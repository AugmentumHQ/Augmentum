/* ==========================================================================
   Augmentum — Model Manager Module
   Ollama/llama.cpp model pull, delete, load/unload, GGUF browser
   ========================================================================== */

import { dismissToast, escapeHtml, showToast } from './app.js';
import { openSamplingEditor } from './sampling-editor.js';
import {
  deleteEngineModelLoadProfile,
  fetchCapabilities,
  fetchModels,
  addToRecentModels,
  getRecentModels,
  getCapabilities,
  getEngineModelLoadProfile,
  getSettings,
  openProjectorPairer,
  pushPrimaryChatModel,
  save,
  saveEngineModelLoadProfile,
  updateThinkingToggleUI,
  waitForUiSettingsReady,
} from './settings.js';
import { closeImagePanel } from './image.js';
import { getAllModels, invalidate as invalidateModelCache } from './model-cache.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let modalEl = null;
let pullAbortController = null;
let lcppRefreshTimer = null;
let downloadsRefreshTimer = null;
let _prevActiveDownloadIds = new Set();  // track running downloads → refresh list when one finishes
let hfSearchTimer = null;
let hfSearchController = null;
// Last HF search {results, backend}, cached so the file picker's "Back to
// results" button can re-render without a re-fetch or the user re-typing.
let _lastHfSearch = null;
let modelManagerMediaQuery = null;

function emptyModelEntries() {
  return {
    engine: [],
    ollama: [],
    llamacpp: [],
  };
}

// Error-vs-empty discipline: a failed fetch must never render as an empty
// state ("No downloads yet", "No files found", blank pane) — the user can't
// tell "there is nothing" from "the backend is down". Render a visually
// distinct error block with a retry affordance instead. Mirrors the
// backend rule in CLAUDE.md (no silent 200s / no log.debug for failures
// the user needs to know about).
function fetchErrorHtml(what, detail = '') {
  const detailHtml = detail
    ? `<div class="mm-fetch-error-detail">${escapeHtml(String(detail).slice(0, 200))}</div>`
    : '';
  return `
    <div class="mm-fetch-error" style="padding:var(--space-sm);border:1px solid var(--error);border-radius:6px;">
      <div style="font-size:var(--text-xs);color:var(--error);">⚠ Couldn't load ${escapeHtml(what)} — this is a fetch failure, not an empty list.</div>
      ${detailHtml}
      <button class="btn btn-sm mm-fetch-retry" style="margin-top:var(--space-xs);">Retry</button>
    </div>`;
}

function renderFetchError(el, what, detail, retryFn) {
  if (!el) return;
  el.innerHTML = fetchErrorHtml(what, detail);
  const btn = el.querySelector('.mm-fetch-retry');
  if (btn && typeof retryFn === 'function') {
    btn.addEventListener('click', () => retryFn());
  }
}

let modalState = {
  inventory: null,
  engineStatus: null,
  lcppStatus: null,
  roleOptions: [],
  managedModels: [],
  modelEntries: emptyModelEntries(),
  downloads: [],
  activePane: 'overview',
  engineModelCatalog: null,
  engineLoadSheet: {
    modelName: '',
    modelPath: '',
    source: 'manager',
    // Which engine slot this load targets: 'A' (primary chat engine),
    // 'B' (secondary resident chat model), 'C' (classifier/utility/vision).
    // Defaults to 'A' so every pre-existing caller behaves exactly as before.
    slot: 'A',
  },
};

// Per-slot metadata for the load sheet + the header picker's A|B|C control.
// One table so the label, endpoint, and role wording can never drift apart
// across the three call sites that need them.
export const ENGINE_SLOTS = {
  A: {
    label: 'Slot A — primary',
    eyebrow: 'Built-in Engine Load Setup',
    copy: 'Your chat model. Saved defaults are reused whenever you pick this model from the header.',
    endpoint: '/api/engine/v2/models/load',
    modelField: 'model',
  },
  B: {
    label: 'Slot B — utility',
    eyebrow: 'Slot B Load Setup (utility role)',
    copy: 'Handles utility work — chat titles, memory consolidation, compaction, reflection, the narrative distiller — on its own port, so it never displaces your chat model. Also usable as a second chat model you can pin per-conversation. Leave it empty and utility work falls back to your primary.',
    endpoint: '/api/engine/v2/secondary/load',
    modelField: 'model_path',
  },
  C: {
    label: 'Slot C — classifier',
    eyebrow: 'Slot C Load Setup (classifier / vision)',
    copy: 'The small resident workhorse behind the classifier and vision roles. Stays loaded (idle timeout 0) so voice and the architect keep their latency budget. Utility work lives in Slot B and no longer queues behind this.',
    endpoint: '/api/engine/v2/classifier/load',
    modelField: 'model_path',
  },
};

// DOM refs inside the modal (resolved on first open)
let dom = {};

// Suggestion chips per backend
// Note: Docker Model Runner (DMR) uses the Ollama API, so these chips work
// for DMR too. DMR models use an ai/ prefix (e.g. ai/qwen2.5-coder) but the
// Ollama-compatible /api/tags endpoint reports them with their full names.
const ollamaChips = [
  // Tier 1: Best all-rounders (most users start here)
  { label: 'Qwen 3.5 7B', model: 'qwen3.5:7b' },
  { label: 'Gemma 3 12B', model: 'gemma3:12b' },
  { label: 'Llama 4 Scout', model: 'llama4:scout' },
  // Tier 2: Coding + reasoning
  { label: 'Qwen 2.5 Coder 14B', model: 'qwen2.5-coder:14b' },
  { label: 'DeepSeek R1 14B', model: 'deepseek-r1:14b' },
  { label: 'GLM 4 9B', model: 'glm4:9b' },
  { label: 'Cogito 14B', model: 'cogito:14b' },
  // Tier 3: Smaller / lighter
  { label: 'Qwen 3.5 4B', model: 'qwen3.5:4b' },
  { label: 'Gemma 3 4B', model: 'gemma3:4b' },
  { label: 'Phi 4 Mini', model: 'phi4-mini' },
  // Tier 4: Larger
  { label: 'Qwen 3.5 32B', model: 'qwen3.5:32b' },
  { label: 'Gemma 3 27B', model: 'gemma3:27b' },
  { label: 'Mistral Small 3.2', model: 'mistral-small3.2' },
];

const llamacppChips = [
  { label: 'Qwen 3.6 27B', model: 'unsloth/Qwen3.6-27B-GGUF' },
  { label: 'Qwen 3.6 35B A3B', model: 'unsloth/Qwen3.6-35B-A3B-GGUF' },
  { label: 'Gemma 4 E4B', model: 'unsloth/gemma-4-E4B-it-GGUF' },
  { label: 'Magistral Small 2507', model: 'unsloth/Magistral-Small-2507-GGUF' },
  { label: 'Phi 4', model: 'unsloth/phi-4-GGUF' },
  { label: 'GLM 4.7 Flash', model: 'unsloth/GLM-4.7-Flash-GGUF' },
  { label: 'Gemma 4 31B', model: 'unsloth/gemma-4-31B-it-GGUF' },
  { label: 'Phi 4 Mini', model: 'unsloth/Phi-4-mini-instruct-GGUF' },
  { label: 'Gemma 4 E2B', model: 'unsloth/gemma-4-E2B-it-GGUF' },
  { label: 'Qwen 2.5 Coder 14B', model: 'unsloth/Qwen2.5-Coder-14B-Instruct-128K-GGUF' },
  { label: 'DeepSeek R1 14B', model: 'unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF' },
  { label: 'Llama 3.3 70B', model: 'unsloth/Llama-3.3-70B-Instruct-GGUF' },
];

// Engine chips come from the friendly download catalog, not installed files.
let engineChips = [
  { label: 'Qwen 3.6 27B', model: 'qwen3.6-27b:ud_q4_k_xl' },
  { label: 'Qwen 3.6 35B A3B', model: 'qwen3.6-35b-a3b:ud_q4_k_xl' },
  { label: 'Gemma 4 E4B', model: 'gemma-4-e4b-it:ud_q4_k_xl' },
  { label: 'Magistral Small 2507', model: 'magistral-small-2507:ud_q4_k_xl' },
  { label: 'Phi 4', model: 'phi-4:q4_k_m' },
  { label: 'GLM 4.7 Flash', model: 'glm-4.7-flash:ud_q4_k_xl' },
  { label: 'Gemma 4 31B', model: 'gemma-4-31b-it:ud_q4_k_xl' },
  { label: 'Phi 4 Mini', model: 'phi-4-mini:q4_k_m' },
  { label: 'Gemma 4 E2B', model: 'gemma-4-e2b-it:ud_q4_k_xl' },
  { label: 'Qwen 2.5 Coder 14B', model: 'qwen2.5-coder-14b:q4_k_m' },
  { label: 'DeepSeek R1 14B', model: 'deepseek-r1-14b:q4_k_m' },
  { label: 'Llama 3.3 70B', model: 'llama-3.3-70b:ud_q4_k_xl' },
];

const engineRecommendationMeta = {
  'qwen3.6-27b': {
    label: 'Qwen 3.6 27B',
    blurb: 'A strong all-around local flagship for coding, chat, and long-context work.',
    fit: 'High-end',
    tags: ['Coding', 'General'],
  },
  'qwen3.6-35b-a3b': {
    label: 'Qwen 3.6 35B A3B',
    blurb: 'A newer MoE Qwen option when you want more capability without jumping to a 70B class model.',
    fit: 'Heavy',
    tags: ['Coding', 'Reasoning'],
  },
  'gemma-4-e4b-it': {
    label: 'Gemma 4 E4B',
    blurb: 'A compact Gemma 4 pick with a much lighter footprint for everyday local use.',
    fit: 'Great fit',
    tags: ['Chat', 'Light'],
  },
  'magistral-small-2507': {
    label: 'Magistral Small 2507',
    blurb: 'A polished reasoning-oriented Mistral model that still fits ambitious home setups.',
    fit: 'Heavy',
    tags: ['Reasoning', 'Quality'],
  },
  'phi-4': {
    label: 'Phi 4',
    blurb: 'A compact reasoning model that stays practical on smaller machines.',
    fit: 'Balanced',
    tags: ['Reasoning', 'Compact'],
  },
  'glm-4.7-flash': {
    label: 'GLM 4.7 Flash',
    blurb: 'A newer efficient MoE model for strong capability with a lighter active footprint.',
    fit: 'Heavy',
    tags: ['General', 'Efficient'],
  },
  'gemma-4-31b-it': {
    label: 'Gemma 4 31B',
    blurb: 'A workstation-class Gemma 4 option when you want a major quality jump locally.',
    fit: 'Workstation',
    tags: ['Chat', 'Large'],
  },
  'qwen2.5-coder-14b': {
    label: 'Qwen 2.5 Coder 14B',
    blurb: 'A strong coding-first pick for edits, tools, and project work.',
    tags: ['Coding', 'Tools'],
  },
  'deepseek-r1-14b': {
    label: 'DeepSeek R1 14B',
    blurb: 'A heavier reasoning-oriented option for slower, more deliberate work.',
    fit: 'Balanced',
    tags: ['Reasoning', 'Tools'],
  },
  'phi-4-mini': {
    label: 'Phi 4 Mini',
    blurb: 'A tiny option for background helpers and quick local prompts.',
    fit: 'Lightweight',
    tags: ['Light', 'Utility'],
  },
  'gemma-4-e2b-it': {
    label: 'Gemma 4 E2B',
    blurb: 'A very small Gemma 4 option for machines that need the lightest modern baseline.',
    fit: 'Lightweight',
    tags: ['Chat', 'Small'],
  },
  'llama-3.3-70b': {
    label: 'Llama 3.3 70B',
    blurb: 'A workstation-class model for ambitious local setups.',
    tags: ['Premium', 'Workstation'],
  },
  'qwen3.5-7b': {
    label: 'Qwen 3.5 7B',
    blurb: 'An older but still practical fallback if you want a smaller Qwen baseline.',
    tags: ['Chat', 'Legacy'],
  },
  'qwen3.5-4b': {
    label: 'Qwen 3.5 4B',
    blurb: 'A lighter legacy Qwen option for utility work and older machines.',
    tags: ['Light', 'Legacy'],
  },
  'gemma-3-12b': {
    label: 'Gemma 3 12B',
    blurb: 'A previous-generation Gemma option that still works well when already familiar.',
    tags: ['Chat', 'Legacy'],
  },
  'gemma-3-4b': {
    label: 'Gemma 3 4B',
    blurb: 'A legacy compact Gemma model for low-memory setups.',
    tags: ['Light', 'Legacy'],
  },
  'llama-4-scout': {
    label: 'Llama 4 Scout',
    blurb: 'A legacy large-model option kept available for manual pulls.',
    tags: ['Large', 'Legacy'],
  },
  'cogito-14b': {
    label: 'Cogito 14B',
    blurb: 'A legacy generalist option kept available for manual pulls.',
    tags: ['General', 'Legacy'],
  },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1000));
  const val = bytes / Math.pow(1000, i);
  return val.toFixed(i > 1 ? 1 : 0) + ' ' + units[i];
}

function formatModelFileSize(bytes, fallback = 'Size pending') {
  return bytes > 0 ? formatBytes(bytes) : fallback;
}

function setDeterminateProgress(pct) {
  const fill = q('mm-progress-fill');
  fill.classList.remove('mm-indeterminate');
  fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function setIndeterminateProgress() {
  const fill = q('mm-progress-fill');
  fill.classList.add('mm-indeterminate');
  fill.style.width = '35%';
}

const q = (id) => dom[id] || (dom[id] = modalEl.querySelector(`#${id}`));

function formatCount(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function prettifyEngineCatalogLabel(name) {
  const key = String(name || '').trim().toLowerCase();
  const metaLabel = engineRecommendationMeta[key]?.label;
  if (metaLabel) return metaLabel;
  return key
    .replace(/-/g, ' ')
    .replace(/\b(\d+(?:\.\d+)?)b\b/gi, (_, size) => `${size}B`)
    .replace(/\b([a-z])/g, (match) => match.toUpperCase());
}

// Best-effort check for whether a chip/search-result corresponds to a
// model already on disk. Match strategy: take the bare name (everything
// before ':' for engine/ollama IDs, the repo path for HF IDs) and
// substring-test against installed model names case-insensitively.
//
// Examples:
//   chip "qwen3.6-27b:ud_q4_k_xl"  → key "qwen3.6-27b"
//   chip "unsloth/Qwen3.6-27B-GGUF"→ key "qwen3.6-27b" (also matches its installs)
//   installed "qwen3.6-27b-ud_q4_k_xl.gguf" → contains "qwen3.6-27b" ✓
//
// Imperfect (could false-positive on "qwen2.5" matching a chip for a
// different family in rare cases), but the cost of a false positive
// (user opens picker, sees "already there") is much lower than a false
// negative (user accidentally re-downloads 14GB).
function isModelInstalled(modelId) {
  if (!modelId) return false;
  const installed = (modalState.managedModels || [])
    .map((m) => String(m.name || '').toLowerCase());
  if (installed.length === 0) return false;

  // Strip backend tag suffix and HF org prefix to get a comparison key
  let key = String(modelId).toLowerCase();
  if (key.includes(':')) key = key.split(':')[0];
  if (key.includes('/')) key = key.split('/').pop();
  // Drop common suffixes like "-gguf" so HF repo names match installed bare names
  key = key.replace(/-gguf$/i, '').replace(/\.gguf$/i, '');
  if (key.length < 3) return false;

  return installed.some((name) => name.includes(key));
}

function getBackendProfile(backend) {
  const isDmr = getCapabilities().is_docker_model_runner;
  const profiles = {
    engine: {
      label: 'Built-in Engine',
      shortLabel: 'Engine',
      description: 'Keep GGUF models inside Augmentum for a simple all-in-one setup.',
      help: 'Best when you want Augmentum to manage local GGUF files directly.',
      placeholder: 'e.g. qwen3.6-27b:ud_q4_k_xl',
      browseLabel: 'Browse GGUF models on HuggingFace',
      browseUrl: 'https://huggingface.co/models?sort=trending&search=gguf',
      actionLabel: 'Add models here',
      selectionNote: 'Augmentum handles the model files and loading for you.',
    },
    ollama: {
      label: isDmr ? 'Docker Model Runner' : 'Ollama',
      shortLabel: isDmr ? 'Docker' : 'Ollama',
      description: isDmr
        ? 'Uses Docker-managed models that show up in Augmentum automatically.'
        : 'One-click downloads and simple model switching through Ollama.',
      help: isDmr
        ? 'This setup reads models from Docker Model Runner.'
        : 'Great for quick installs and easy local model management.',
      placeholder: 'e.g. llama3.1:8b',
      browseLabel: 'Browse models on ollama.com',
      browseUrl: 'https://ollama.com/search',
      actionLabel: 'Use this device',
      selectionNote: isDmr
        ? 'Model installs happen in Docker, then they appear here.'
        : 'Downloads go straight into your Ollama library.',
    },
    llamacpp: {
      label: 'llama.cpp',
      shortLabel: 'llama.cpp',
      description: 'Great for hand-picked GGUF files and advanced local tuning.',
      help: 'Choose a GGUF repository, then pick the exact file you want to download.',
      placeholder: 'e.g. bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
      browseLabel: 'Browse GGUF models on HuggingFace',
      browseUrl: 'https://huggingface.co/models?sort=trending&search=gguf',
      actionLabel: 'Add models here',
      selectionNote: 'Best if you want to pick a specific GGUF quantization.',
    },
    vllm: {
      label: 'vLLM Engine',
      shortLabel: 'vLLM',
      description: 'Run safetensors models — new architectures + parallel workflows on a big GPU.',
      help: 'Search safetensors models; the whole repo downloads and is served by the vLLM engine.',
      placeholder: 'e.g. Qwen/Qwen3-Coder-30B-A3B-Instruct',
      browseLabel: 'Browse safetensors models on HuggingFace',
      browseUrl: 'https://huggingface.co/models?sort=trending&library=safetensors',
      actionLabel: 'Add models here',
      selectionNote: 'Downloads the full repo to a drive you pick, served by the vLLM engine.',
    },
  };

  return profiles[backend] || {
    label: backend,
    shortLabel: backend,
    description: '',
    help: '',
    placeholder: '',
    browseLabel: 'Browse models',
    browseUrl: '#',
    actionLabel: 'Select',
    selectionNote: '',
  };
}

function sortModelsForDisplay(models) {
  return [...models].sort((a, b) => {
    const aSelected = a.isSelected ? 1 : 0;
    const bSelected = b.isSelected ? 1 : 0;
    if (aSelected !== bSelected) return bSelected - aSelected;

    const aLoaded = a.isLoaded ? 1 : 0;
    const bLoaded = b.isLoaded ? 1 : 0;
    if (aLoaded !== bLoaded) return bLoaded - aLoaded;

    return a.model.name.localeCompare(b.model.name);
  });
}

function recentManagedModelNames(models) {
  const available = new Set((models || []).map((model) => model?.name).filter(Boolean));
  const seen = new Set();
  const names = [];
  for (const name of getRecentModels()) {
    if (!available.has(name) || seen.has(name)) continue;
    seen.add(name);
    names.push(name);
    if (names.length >= 8) break;
  }
  return names;
}

function isSelectableModelName(name) {
  return !!name && !name.startsWith('a/') && !name.startsWith('n/') && !name.startsWith('p/');
}

function buildRoleOptions(models) {
  const names = Array.from(new Set(
    (models || [])
      .map((model) => model?.name || '')
      .filter(isSelectableModelName)
  ));

  names.sort((a, b) => a.localeCompare(b));
  return names;
}

function formatTokenCount(count) {
  const value = Number(count || 0);
  if (!value) return 'Auto';
  if (value >= 1000) {
    const short = value % 1000 === 0 ? (value / 1000).toFixed(0) : (value / 1000).toFixed(1);
    return `${short}K`;
  }
  return `${value}`;
}

function formatIdleTimeout(seconds) {
  const value = Number(seconds || 0);
  if (value <= 0) return 'Stays loaded';
  if (value % 3600 === 0) return `${value / 3600}h idle`;
  if (value % 60 === 0) return `${value / 60}m idle`;
  return `${value}s idle`;
}

function normalizeEngineProfile(profile = {}, modelMaxContext = 0) {
  const maxCtx = Number(modelMaxContext || 0) || Number(profile.ctx_size || 0) || 8192;
  const ctx = Math.max(2048, Math.min(Number.parseInt(profile.ctx_size, 10) || Math.min(32384, maxCtx), maxCtx));
  const batchSize = Math.max(32, Math.min(Number.parseInt(profile.batch_size, 10) || 512, 8192));
  const gpuLayersMode = ['auto', 'cpu', 'custom', 'moe_cpu', 'moe_first_n_cpu', 'moe_auto_vram'].includes(String(profile.gpu_layers_mode || 'auto'))
    ? String(profile.gpu_layers_mode || 'auto')
    : 'auto';
  const gpuLayers = Math.max(0, Number.parseInt(profile.gpu_layers, 10) || 0);
  // MoE expert-offload layer count — only meaningful when
  // gpu_layers_mode === 'moe_first_n_cpu' or 'moe_auto_vram'. Three
  // distinct signals: explicit positive N, explicit 0 (all experts on
  // GPU — fits for tiny MoEs), or unset/blank (let backend autofit
  // pick — safe-all-CPU default for big MoEs, autofit-balanced for
  // moe_auto_vram mode). Returning ``undefined`` here removes the
  // field from the wire payload entirely via JSON.stringify, which is
  // semantically distinct from sending 0. The prior ``|| 16``
  // fallback committed 16-layers-on-CPU for blank inputs, which OOMs
  // any model with a ≥40 GB expert pool (Qwen3.5-122B-A10B-class).
  const moeCpuLayersRaw = Number.parseInt(profile.moe_cpu_layers, 10);
  const moeCpuLayers = Number.isFinite(moeCpuLayersRaw)
    ? Math.max(0, Math.min(moeCpuLayersRaw, 999))
    : undefined;
  const kvCacheType = String(profile.kv_cache_type || '');
  // ``0`` is a MEANINGFUL value (never idle-unload — keep resident), so we
  // must not use ``|| 600``: ``0 || 600`` evaluates to 600, making idle=0
  // impossible to save. Preserve an explicit finite 0; only fall back to the
  // 600 s default when the field is blank/NaN. (Same NaN-aware pattern as
  // ``moe_cpu_layers`` / ``seed`` above.)
  const idleRaw = Number.parseInt(profile.idle_timeout, 10);
  const idleTimeout = Number.isFinite(idleRaw) ? Math.max(0, idleRaw) : 600;
  const chatTemplateMode = ['embedded', 'builtin', 'custom'].includes(String(profile.chat_template_mode || 'embedded'))
    ? String(profile.chat_template_mode || 'embedded')
    : 'embedded';
  const reasoningFormat = ['', 'deepseek', 'none', 'auto'].includes(String(profile.reasoning_format || ''))
    ? String(profile.reasoning_format || '')
    : '';
  // Validate JSON for chat_template_kwargs early so we don't ship a malformed
  // string to llama-server (it would error out on startup).
  let chatTemplateKwargs = String(profile.chat_template_kwargs || '').trim();
  if (chatTemplateKwargs) {
    try { JSON.parse(chatTemplateKwargs); } catch { chatTemplateKwargs = ''; }
  }
  // V-cache override. Empty string means "use K's value" — preserves
  // backward compat for saved profiles that pre-date this field.
  const kvCacheTypeV = String(profile.kv_cache_type_v || '');
  // CPU thread pools. ``0`` = let llama-server pick the default
  // (half of available hardware threads). Same shape on both pools.
  const cpuThreads = Math.max(0, Math.min(Number.parseInt(profile.cpu_threads, 10) || 0, 256));
  const cpuThreadsBatch = Math.max(0, Math.min(Number.parseInt(profile.cpu_threads_batch, 10) || 0, 256));
  const mlock = profile.mlock === true || profile.mlock === 'true';
  // Sampler seed. ``-1`` (or any negative) = random per request. Kept
  // as a plain number so the existing field-change → plan-refresh
  // pipeline picks it up like any other knob.
  const rawSeed = Number.parseInt(profile.seed, 10);
  const seed = Number.isFinite(rawSeed) ? Math.max(-1, Math.min(rawSeed, 2147483647)) : -1;
  // Multi-GPU placement. Empty/0/'' means "let llama-server pick" —
  // the UI hides these inputs on single-GPU hosts but the values still
  // round-trip through the saved profile (so a profile authored on a
  // multi-GPU machine doesn't silently lose its placement preferences
  // when opened on a single-GPU host).
  const tensorSplit = String(profile.tensor_split || '').trim();
  const mainGpu = Math.max(0, Math.min(Number.parseInt(profile.main_gpu, 10) || 0, 15));
  const splitMode = ['', 'layer', 'row', 'none'].includes(String(profile.split_mode || ''))
    ? String(profile.split_mode || '')
    : '';
  const normalized = {
    ctx_size: ctx,
    batch_size: batchSize,
    gpu_layers_mode: gpuLayersMode,
    gpu_layers: gpuLayers,
    moe_cpu_layers: moeCpuLayers,
    kv_cache_type: ['', 'f16', 'q8_0', 'q4_0'].includes(kvCacheType) ? kvCacheType : '',
    kv_cache_type_v: ['', 'f16', 'q8_0', 'q4_0'].includes(kvCacheTypeV) ? kvCacheTypeV : '',
    flash_attn: profile.flash_attn !== false,
    idle_timeout: idleTimeout,
    cpu_threads: cpuThreads,
    cpu_threads_batch: cpuThreadsBatch,
    mlock,
    seed,
    tensor_split: tensorSplit,
    main_gpu: mainGpu,
    split_mode: splitMode,
    chat_template_mode: chatTemplateMode,
    chat_template_kwargs: chatTemplateKwargs,
    reasoning_format: reasoningFormat,
  };
  // LoRA adapter: stored only when a path is set (parallel to the
  // draft_model pattern). Keeps saved profiles tidy — no stale
  // scale=1.0 entries on profiles that never opted in.
  const loraModel = String(profile.lora_model || '').trim();
  if (loraModel) {
    normalized.lora_model = loraModel;
    const rawScale = Number.parseFloat(profile.lora_scale);
    normalized.lora_scale = Number.isFinite(rawScale)
      ? Math.max(0, Math.min(rawScale, 2.0))
      : 1.0;
  }
  // Only echo back the custom Jinja content when the mode actually needs it,
  // so we don't bloat saved profiles with stale textareas.
  if (chatTemplateMode === 'custom' && profile.chat_template_content) {
    normalized.chat_template_content = String(profile.chat_template_content);
  }
  const draftModel = String(profile.draft_model || '').trim();
  if (draftModel) {
    normalized.draft_model = draftModel;
    normalized.draft_max = Math.max(1, Math.min(Number.parseInt(profile.draft_max, 10) || 5, 32));
    normalized.draft_ctx_size = Math.max(512, Math.min(Number.parseInt(profile.draft_ctx_size, 10) || 2048, 32768));
    // ``0`` is meaningful for both (draft fully on CPU; min draft length 0),
    // so preserve an explicit finite 0 rather than coalescing it to the
    // default via ``|| N``. The populate side already uses ``?? default``;
    // this makes the save side match.
    const draftGpuRaw = Number.parseInt(profile.draft_gpu_layers, 10);
    normalized.draft_gpu_layers = Number.isFinite(draftGpuRaw)
      ? Math.max(0, Math.min(draftGpuRaw, 999))
      : 999;
    const draftMinRaw = Number.parseInt(profile.draft_min, 10);
    normalized.draft_min = Number.isFinite(draftMinRaw)
      ? Math.max(0, Math.min(draftMinRaw, 32))
      : 1;
    const pMin = Number.parseFloat(profile.draft_p_min);
    normalized.draft_p_min = Number.isFinite(pMin) ? Math.max(0, Math.min(pMin, 1)) : 0.75;
  }
  // MTP self-speculation per-load. 'auto' (or unset) means "inherit
  // engine_mtp_enabled" → omit the key so the backend falls back to
  // the global. Explicit true/false stays in the profile.
  const mtpRaw = profile.mtp_enabled;
  if (mtpRaw === true || mtpRaw === 'true') {
    normalized.mtp_enabled = true;
  } else if (mtpRaw === false || mtpRaw === 'false') {
    normalized.mtp_enabled = false;
  }
  const mtpNMaxRaw = Number.parseInt(profile.mtp_n_max, 10);
  if (Number.isFinite(mtpNMaxRaw)) {
    normalized.mtp_n_max = Math.max(1, Math.min(mtpNMaxRaw, 16));
  }
  // Vision (mmproj) per-load. Same tri-state shape as MTP. 'auto' omits
  // the key so backend falls back to engine_auto_pair_mmproj.
  const visionRaw = profile.vision_mode;
  if (visionRaw === true || visionRaw === 'true') {
    normalized.vision_mode = true;
  } else if (visionRaw === false || visionRaw === 'false') {
    normalized.vision_mode = false;
  }
  return normalized;
}

function canonicalEngineModelRef(value = '') {
  return String(value || '').trim().replace(/\\/g, '/');
}

function formatEngineModelRef(value = '') {
  const clean = canonicalEngineModelRef(value);
  if (!clean) return '';
  const last = clean.split('/').pop() || clean;
  return last.toLowerCase().endsWith('.gguf') ? last.slice(0, -5) : last;
}

function engineProfileSummary(profile = {}) {
  const parts = [];
  if (profile.ctx_size) parts.push(`${formatTokenCount(profile.ctx_size)} ctx`);
  if (profile.gpu_layers_mode === 'cpu') parts.push('CPU only');
  else if (profile.gpu_layers_mode === 'custom') parts.push(`${profile.gpu_layers || 0} GPU layers`);
  else if (profile.gpu_layers_mode === 'moe_cpu') parts.push('MoE: experts on CPU');
  else if (profile.gpu_layers_mode === 'moe_first_n_cpu') {
    // Distinguish unset (let backend autofit pick) from explicit 0
    // (all experts on GPU). Using ``== null`` covers both null and
    // undefined; an explicit numeric 0 falls through to the labelled
    // branch.
    parts.push(profile.moe_cpu_layers == null
      ? 'MoE: experts autofit'
      : `MoE: first ${profile.moe_cpu_layers} on CPU`);
  }
  else if (profile.gpu_layers_mode === 'moe_auto_vram') parts.push('MoE: VRAM-balanced');
  else parts.push('Auto GPU');
  if (profile.kv_cache_type) parts.push(`${profile.kv_cache_type.toUpperCase()} KV`);
  if (profile.draft_model) parts.push(`Draft ${formatEngineModelRef(profile.draft_model)}`);
  if (profile.idle_timeout != null) parts.push(formatIdleTimeout(profile.idle_timeout));
  return parts.join(' · ');
}

function renderRoleSettings() {
  const utilitySelect = q('mm-role-utility');
  const classifierSelect = q('mm-role-classifier');
  const heavyweightSelect = q('mm-role-heavyweight');
  // Game-agent lane pins. They live here rather than in a game-only panel
  // because they are the same KIND of decision as the roles above — and
  // because until now they had no UI at all, so an unpinned lane silently
  // resolved to whichever model happened to be first on the default backend.
  const gamePlannerSelect = q('mm-game-planner');
  const gameFastSelect = q('mm-game-fast');
  const visualVerifySelect = q('mm-visual-verify');
  if (!utilitySelect || !classifierSelect) return;

  const settings = getSettings();
  const available = [...(modalState.roleOptions || [])];
  const utilityValue = settings.utilityModel || '';
  const classifierValue = settings.classifierModel || '';
  const heavyweightValue = settings.heavyweightModel || '';
  const gamePlannerValue = settings.gameAgentPlannerModel || '';
  const gameFastValue = settings.gameAgentFastModel || '';
  const visualVerifyValue = settings.coderVisualVerifyModel || '';

  if (utilityValue && !available.includes(utilityValue)) available.unshift(utilityValue);
  if (classifierValue && !available.includes(classifierValue)) available.unshift(classifierValue);
  if (heavyweightValue && !available.includes(heavyweightValue)) available.unshift(heavyweightValue);
  if (gamePlannerValue && !available.includes(gamePlannerValue)) available.unshift(gamePlannerValue);
  if (gameFastValue && !available.includes(gameFastValue)) available.unshift(gameFastValue);
  if (visualVerifyValue && !available.includes(visualVerifyValue)) available.unshift(visualVerifyValue);

  const renderOptions = (select, currentValue, emptyLabel) => {
    select.innerHTML = [
      `<option value="">${escapeHtml(emptyLabel)}</option>`,
      ...available.map((name) => {
        const unavailable = !modalState.roleOptions?.includes(name);
        const label = unavailable ? `${name} (currently unavailable)` : name;
        return `<option value="${escapeHtml(name)}">${escapeHtml(label)}</option>`;
      }),
    ].join('');
    select.value = currentValue || '';
  };

  renderOptions(utilitySelect, utilityValue, 'Auto - use Primary');
  renderOptions(classifierSelect, classifierValue, 'Auto - use Utility');
  if (heavyweightSelect) {
    renderOptions(heavyweightSelect, heavyweightValue, 'Not configured');
  }
  // Verify toggle rides with the heavyweight pin — it's only meaningful when
  // one is set. Default on (undefined → checked).
  const verifyToggle = q('mm-verify-enabled');
  if (verifyToggle) {
    verifyToggle.checked = settings.coderVerifyEnabled !== false;
    verifyToggle.onchange = () => persistRoleSettings(snapshot());
  }
  // The empty labels name the ROLE each lane falls through to, not "auto" in
  // the abstract — the fallback is now a declared role, so the picker can say
  // exactly where an unset pin lands instead of leaving the user to find out.
  if (gamePlannerSelect) renderOptions(gamePlannerSelect, gamePlannerValue, 'Auto - use Primary');
  if (gameFastSelect) renderOptions(gameFastSelect, gameFastValue, 'Auto - use Classifier');
  if (visualVerifySelect) renderOptions(visualVerifySelect, visualVerifyValue, 'Auto - current routing');

  // Snapshot all of them on every change so a write reflects the
  // user's full current picker state (saves a round-trip if they
  // change two in quick succession).
  const snapshot = () => ({
    utilityModel: utilitySelect.value.trim(),
    classifierModel: classifierSelect.value.trim(),
    heavyweightModel: heavyweightSelect ? heavyweightSelect.value.trim() : (settings.heavyweightModel || ''),
    coderVerifyEnabled: verifyToggle ? verifyToggle.checked : (settings.coderVerifyEnabled !== false),
    gameAgentPlannerModel: gamePlannerSelect ? gamePlannerSelect.value.trim() : (settings.gameAgentPlannerModel || ''),
    gameAgentFastModel: gameFastSelect ? gameFastSelect.value.trim() : (settings.gameAgentFastModel || ''),
    coderVisualVerifyModel: visualVerifySelect ? visualVerifySelect.value.trim() : (settings.coderVisualVerifyModel || ''),
  });

  utilitySelect.onchange = () => persistRoleSettings(snapshot());
  classifierSelect.onchange = () => persistRoleSettings(snapshot());
  if (heavyweightSelect) heavyweightSelect.onchange = () => persistRoleSettings(snapshot());
  if (gamePlannerSelect) gamePlannerSelect.onchange = () => persistRoleSettings(snapshot());
  if (gameFastSelect) gameFastSelect.onchange = () => persistRoleSettings(snapshot());
  if (visualVerifySelect) visualVerifySelect.onchange = () => persistRoleSettings(snapshot());
}

async function persistRoleSettings({
  utilityModel, classifierModel, heavyweightModel,
  coderVerifyEnabled, gameAgentPlannerModel, gameAgentFastModel,
  coderVisualVerifyModel,
}) {
  const settings = getSettings();
  settings.utilityModel = utilityModel || '';
  settings.classifierModel = classifierModel || '';
  settings.heavyweightModel = heavyweightModel || '';
  if (coderVerifyEnabled !== undefined) settings.coderVerifyEnabled = coderVerifyEnabled;
  settings.gameAgentPlannerModel = gameAgentPlannerModel || '';
  settings.gameAgentFastModel = gameAgentFastModel || '';
  settings.coderVisualVerifyModel = coderVisualVerifyModel || '';
  save();

  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        utility_model: settings.utilityModel || '',
        classifier_model: settings.classifierModel || '',
        heavyweight_model: settings.heavyweightModel || '',
        coder_verify_enabled: settings.coderVerifyEnabled !== false,
        // Same endpoint: PUT /api/config/tools writes _TOOL_SETTINGS and
        // _STRING_SETTINGS alike, and both pins are already registered there.
        game_agent_planner_model: settings.gameAgentPlannerModel || '',
        game_agent_fast_model: settings.gameAgentFastModel || '',
        coder_visual_verify_model: settings.coderVisualVerifyModel || '',
      }),
    });
  } catch {
    showToast('Could not save task roles', 'error');
  }

  renderRoleSettings();
}

function getManagedEngineEntries() {
  return modalState.modelEntries?.engine || [];
}

// --- Drive (storage volume) labelling + filtering -------------------------
// Engine models can live across multiple mounted folders (e.g. /models/host,
// /models/spare). We surface which drive each model is on, summed usage per
// drive, and a click-to-filter so the user can tell who's using what space
// without having to ask. The drive "label" is just the mount's folder name,
// which keeps it generic for any OSS install's storage layout.

function engineModelDirs() {
  return modalState.engineModelDirs || [];
}

function hasMultipleDrives() {
  return engineModelDirs().length > 1;
}

function driveLabelFromDir(dir) {
  const clean = String(dir || '').replace(/\/+$/, '');
  if (!clean) return 'disk';
  return clean.split('/').pop() || clean;
}

// Longest-prefix match of a model path against the configured dirs. Returns
// the raw dir (used as the stable filter key), or null if it's outside all
// configured roots.
function driveForPath(path) {
  const p = String(path || '');
  if (!p) return null;
  let best = null;
  for (const dir of engineModelDirs()) {
    const d = String(dir || '').replace(/\/+$/, '');
    if (!d) continue;
    if ((p === d || p.startsWith(`${d}/`)) && (!best || d.length > best.length)) {
      best = d;
    }
  }
  return best;
}

// Map engine model display-name → its drive dir, via the v2 catalog (which is
// the only source carrying each model's on-disk path). Mirrors the name⇄entry
// matching in resolveEngineModelRecord so badges line up with the cards.
function buildDriveByModelName() {
  const map = new Map();
  for (const entry of (modalState.engineModelCatalog || [])) {
    const dir = driveForPath(entry.path);
    const filename = String(entry.filename || '').toLowerCase();
    const stem = filename.endsWith('.gguf') ? filename.slice(0, -5) : filename;
    if (stem) map.set(stem, dir);
    const ref = formatEngineModelRef(entry.path || entry.filename || '').toLowerCase();
    if (ref) map.set(ref, dir);
  }
  return map;
}

function driveForModelName(name, driveMap) {
  const key = String(name || '').toLowerCase();
  return driveMap.has(key) ? driveMap.get(key) : null;
}

// Toggle the active drive filter (clicking the active drive clears it) and
// re-apply it to the already-rendered cards — no full list rebuild needed.
function setDriveFilter(dir) {
  modalState.driveFilter = (modalState.driveFilter === dir) ? null : (dir || null);
  applyDriveFilter();
}

function applyDriveFilter() {
  const list = q('mm-model-list');
  if (!list) return;
  const filter = modalState.driveFilter || null;

  list.querySelectorAll('.mm-model-card').forEach((card) => {
    const drive = card.dataset.drive || '';
    // No filter → show all. Filtering a drive hides cards on other drives,
    // and cards with no drive (non-engine backends aren't on these volumes).
    const show = !filter || drive === filter;
    card.classList.toggle('mm-filtered-out', !show);
  });

  list.querySelectorAll('.mm-model-group').forEach((group) => {
    const anyVisible = group.querySelector('.mm-model-card:not(.mm-filtered-out)');
    group.classList.toggle('mm-filtered-out', !anyVisible);
  });

  list.querySelectorAll('[data-drive-chip]').forEach((chip) => {
    const active = chip.dataset.driveChip === (filter || '__all__');
    chip.classList.toggle('active', active);
  });
  list.querySelectorAll('[data-drive-filter]').forEach((badge) => {
    badge.classList.toggle('active', Boolean(filter) && badge.dataset.driveFilter === filter);
  });
}

function getActiveDownloads() {
  return (modalState.downloads || []).filter((download) => (
    download.status === 'pending' || download.status === 'running'
  ));
}

function getEngineStatePresentation(engineStatus) {
  if (!getCapabilities().has_engine) {
    return {
      label: 'Unavailable',
      tone: 'muted',
      copy: 'The built-in engine is not available on this machine yet.',
    };
  }

  switch (engineStatus?.state) {
    case 'ready':
      return {
        label: 'Ready',
        tone: 'good',
        copy: engineStatus?.model_id
          ? `Currently using ${formatEngineModelRef(engineStatus.model_id)}.`
          : 'The engine is ready for its next model.',
      };
    case 'starting':
      return {
        label: 'Starting',
        tone: 'busy',
        copy: 'A model is loading into memory now.',
      };
    case 'draining':
      return {
        label: 'Switching',
        tone: 'busy',
        copy: 'The engine is swapping models and finishing in-flight work.',
      };
    default:
      return {
        label: 'Idle',
        tone: 'muted',
        copy: 'No model is loaded right now, but your local library is ready.',
      };
  }
}

function getEngineRecommendationPresentation(chip) {
  const key = String(chip?.model || '').split(':')[0].toLowerCase();
  const meta = engineRecommendationMeta[key] || {};
  const sizeMatch = (chip?.label || key).toLowerCase().match(/(\d+(?:\.\d+)?)b/);
  const sizeB = sizeMatch ? Number.parseFloat(sizeMatch[1]) : 0;
  let fit = meta.fit || 'Recommended';
  if (!meta.fit) {
    if (sizeB <= 4) fit = 'Lightweight';
    else if (sizeB <= 8) fit = 'Great fit';
    else if (sizeB <= 14) fit = 'Balanced';
    else if (sizeB <= 32) fit = 'Heavy';
    else if (sizeB > 32) fit = 'Workstation';
  }

  let tags = meta.tags;
  if (!tags || !tags.length) {
    tags = [];
    if (key.includes('coder')) tags.push('Coding');
    else if (key.includes('r1')) tags.push('Reasoning');
    else tags.push('Chat');
    tags.push(sizeB > 14 ? 'Large' : 'Local');
  }

  return {
    fit,
    tags,
    blurb: meta.blurb || 'A curated engine model to get started quickly.',
  };
}

function selectOverviewEngineEntry() {
  const entries = getManagedEngineEntries();
  if (!entries.length) return null;
  return entries.find((entry) => entry.engineLoaded)
    || entries.find((entry) => entry.isSelected)
    || [...entries].sort((left, right) => {
      const leftSize = Number(left.model?.size || 0);
      const rightSize = Number(right.model?.size || 0);
      if (leftSize !== rightSize) return rightSize - leftSize;
      return String(left.model?.name || '').localeCompare(String(right.model?.name || ''));
    })[0]
    || entries[0];
}

function setActiveManagerPane(nextPane, opts = {}) {
  if (!modalEl) return;
  const availablePanes = Array.from(modalEl.querySelectorAll('[data-mm-pane]'))
    .map((pane) => pane.dataset.mmPane);
  const targetPane = availablePanes.includes(nextPane) ? nextPane : 'overview';
  modalState.activePane = targetPane;

  // Auto-detect models dropped into a model dir: watch only while the Library
  // tab is open (cheap live scan), stop otherwise so it never runs in the bg.
  if (targetPane === 'library') startSafetensorsWatch();
  else stopSafetensorsWatch();

  modalEl.querySelectorAll('[data-mm-pane]').forEach((pane) => {
    pane.classList.toggle('hidden', pane.dataset.mmPane !== targetPane);
  });
  modalEl.querySelectorAll('[data-mm-nav]').forEach((button) => {
    const active = button.dataset.mmNav === targetPane;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  });

  if (targetPane === 'advanced' && opts.expandAdvanced) {
    const advanced = q('mm-advanced-controls');
    if (advanced) advanced.open = true;
  }

  // Fabric tab: lazy-load on first activation. Subsequent activations
  // re-render to pick up state changes (peer connections/disconnections,
  // newly-paired peers, etc.). The fabric module owns its own bind-once
  // logic; calling renderFabricTab repeatedly is safe.
  if (targetPane === 'fabric') {
    // Both failure paths below previously logged to console only, leaving
    // the user staring at a BLANK pane (error-vs-empty discipline: a
    // broken module/render must say so on screen, with a retry).
    const fabricFail = (stage) => (err) => {
      console.error(`fabric ${stage} failed:`, err);
      renderFetchError(
        q('mm-fabric-root'), 'the Fabric panel', err?.message,
        () => setActiveManagerPane('fabric'),
      );
    };
    import('./fabric.js').then((mod) => {
      if (typeof mod.renderFabricTab === 'function') {
        // The fabric module itself absorbs network errors and renders a
        // clean "unavailable" state — reaching this catch means something
        // more fundamental broke (render bug, missing DOM).
        mod.renderFabricTab().catch(fabricFail('tab render'));
      }
    }).catch(fabricFail('module load'));
  }

  // Reset scroll on pane switch. Desktop: .mm-main owns the scroll.
  // Mobile: .mm-body still scrolls (single-column layout). Reset both
  // to be defensive — scrollTop on a non-scrolling container is a no-op.
  const body = modalEl.querySelector('.mm-body');
  if (body) body.scrollTop = 0;
  const main = modalEl.querySelector('.mm-main');
  if (main) main.scrollTop = 0;

  if (opts.focusId) {
    requestAnimationFrame(() => {
      q(opts.focusId)?.focus();
    });
  }
}

function showDiscoverPane(backend = 'engine', opts = {}) {
  setActiveManagerPane('discover', opts);
  const select = q('mm-backend-select');
  if (select) {
    select.value = backend;
    updateBackendUI();
  }
  if (opts.focusInput !== false) {
    requestAnimationFrame(() => {
      q('mm-pull-input')?.focus();
    });
  }
}

function renderOverview() {
  if (!modalEl) return;

  const heroEl = q('mm-overview-hero');
  const recommendedEl = q('mm-overview-recommended');
  const currentEl = q('mm-overview-current');
  const rolesEl = q('mm-overview-roles');
  const activityEl = q('mm-overview-activity');
  const libraryEl = q('mm-overview-library');
  if (!heroEl || !recommendedEl || !currentEl || !rolesEl || !activityEl || !libraryEl) return;

  const engineEntries = getManagedEngineEntries();
  const activeDownloads = getActiveDownloads();
  const engineStatus = modalState.engineStatus;
  const inventory = modalState.inventory || {
    counts: { engine: 0 },
  };
  const enginePresentation = getEngineStatePresentation(engineStatus);
  const engineStorage = engineEntries.reduce((sum, entry) => sum + Number(entry.model?.size || 0), 0);
  const currentEngineName = engineStatus?.model_id
    ? formatEngineModelRef(engineStatus.model_id)
    : 'No model loaded';

  // Per-drive storage rollup — only when models span multiple mounted folders.
  // Each row sums the engine model sizes on that drive and shows free space;
  // clicking jumps to the model list filtered to that drive.
  let driveRollupHtml = '';
  if (hasMultipleDrives()) {
    const driveMap = buildDriveByModelName();
    const usedByDir = new Map();
    for (const entry of engineEntries) {
      const d = driveForModelName(entry.model?.name, driveMap);
      if (!d) continue;
      usedByDir.set(d, (usedByDir.get(d) || 0) + Number(entry.model?.size || 0));
    }
    const storageByDir = new Map((modalState.driveStorage || [])
      .map((s) => [String(s.dir || '').replace(/\/+$/, ''), s]));
    const rows = engineModelDirs().map((dir) => {
      const d = String(dir).replace(/\/+$/, '');
      const label = driveLabelFromDir(d);
      const used = usedByDir.get(d) || 0;
      const s = storageByDir.get(d);
      const free = (s && s.free_bytes != null) ? `${formatBytes(s.free_bytes)} free` : '';
      return `
        <button class="mm-drive-row" data-mm-drive-jump="${escapeHtml(d)}" title="Show only models on ${escapeHtml(label)}">
          <span class="mm-drive-row-label">${escapeHtml(label)}</span>
          <span class="mm-drive-row-meta">${escapeHtml(formatBytes(used))} used${free ? ` · ${escapeHtml(free)}` : ''}</span>
        </button>`;
    }).join('');
    driveRollupHtml = `
      <div class="mm-drive-rollup">
        <div class="mm-drive-rollup-title">Storage by drive</div>
        ${rows}
      </div>`;
  }

  heroEl.innerHTML = `
    <div class="mm-overview-hero-card">
      <div class="mm-overview-hero-copy">
        <div class="mm-overview-eyebrow">Built-in Engine</div>
        <div class="mm-overview-hero-head">
          <div class="mm-overview-hero-title">Local models, tuned inside Augmentum.</div>
          <span class="mm-overview-state ${escapeHtml(enginePresentation.tone)}">${escapeHtml(enginePresentation.label)}</span>
        </div>
        <div class="mm-overview-hero-text">${escapeHtml(enginePresentation.copy)}</div>
      </div>
      <div class="mm-overview-stat-grid">
        <div class="mm-overview-stat">
          <span class="mm-overview-stat-label">Installed</span>
          <span class="mm-overview-stat-value">${escapeHtml(formatCount(inventory.counts.engine || 0, 'model'))}</span>
        </div>
        <div class="mm-overview-stat">
          <span class="mm-overview-stat-label">Current model</span>
          <span class="mm-overview-stat-value">${escapeHtml(currentEngineName)}</span>
        </div>
        <div class="mm-overview-stat">
          <span class="mm-overview-stat-label">Storage used</span>
          <span class="mm-overview-stat-value">${escapeHtml(formatBytes(engineStorage))}</span>
        </div>
        <div class="mm-overview-stat">
          <span class="mm-overview-stat-label">Active downloads</span>
          <span class="mm-overview-stat-value">${escapeHtml(String(activeDownloads.length))}</span>
        </div>
      </div>
      ${driveRollupHtml}
      <div class="mm-overview-actions">
        <button class="btn btn-primary" data-mm-select-backend="engine">Find models</button>
        <button class="btn btn-sm" data-mm-open-pane="library">Open library</button>
        <button class="btn btn-sm" data-mm-open-pane="advanced" data-mm-expand-advanced="true" data-mm-focus="mm-engine-add-dir">Add model folder</button>
        <button class="btn btn-sm" data-mm-refresh-setup="true">Refresh setup</button>
      </div>
    </div>
  `;

  const recommendedCards = engineChips.slice(0, 8).map((chip) => {
    const recommendation = getEngineRecommendationPresentation(chip);
    const quant = String(chip.model || '').split(':')[1] || 'default';
    return `
      <div class="mm-rec-card">
        <div class="mm-rec-head">
          <div>
            <div class="mm-rec-title">${escapeHtml(chip.label)}</div>
            <div class="mm-rec-copy">${escapeHtml(recommendation.blurb)}</div>
          </div>
          <span class="mm-rec-fit">${escapeHtml(recommendation.fit)}</span>
        </div>
        <div class="mm-rec-tags">
          ${recommendation.tags.map((tag) => `<span class="mm-rec-tag">${escapeHtml(tag)}</span>`).join('')}
          <span class="mm-rec-tag">Default ${escapeHtml(quant.toUpperCase())}</span>
        </div>
        <button class="btn btn-sm btn-primary" data-mm-engine-chip="${escapeHtml(chip.model)}">Download</button>
      </div>
    `;
  }).join('');
  recommendedEl.innerHTML = recommendedCards || '<div class="mm-empty">Recommended engine models will appear here.</div>';

  const currentEntry = selectOverviewEngineEntry();
  if (!currentEntry) {
    currentEl.innerHTML = `
      <div class="mm-overview-empty">
        <div class="mm-overview-empty-title">No engine models yet</div>
        <div class="mm-overview-empty-copy">Start with a recommended model and Augmentum can manage the rest.</div>
        <button class="btn btn-primary btn-sm" data-mm-select-backend="engine">Find engine models</button>
      </div>
    `;
  } else {
    currentEl.innerHTML = `
      <div class="mm-overview-card-copy">
        ${currentEntry.engineLoaded
          ? 'This is the model currently active in the built-in engine.'
          : 'This is the next engine model ready to load with its saved defaults.'}
      </div>
      <div id="mm-overview-current-card"></div>
    `;
    const cardMount = currentEl.querySelector('#mm-overview-current-card');
    if (cardMount) {
      const card = createModelCard(
        currentEntry.model,
        currentEntry.isLoaded,
        currentEntry.isOllama,
        currentEntry.routerStatus,
        currentEntry.runningInfo,
        currentEntry.isEngine,
        currentEntry.engineLoaded,
        currentEntry.engineState,
      );
      card.classList.add('mm-overview-current-card');
      cardMount.replaceChildren(card);
    }
  }

  const settings = getSettings();
  rolesEl.innerHTML = `
    <div class="mm-overview-mini-grid">
      <div class="mm-overview-mini-card">
        <div class="mm-overview-mini-label">Utility</div>
        <div class="mm-overview-mini-value">${escapeHtml(settings.utilityModel || 'Auto - use Primary')}</div>
      </div>
      <div class="mm-overview-mini-card">
        <div class="mm-overview-mini-label">Classifier</div>
        <div class="mm-overview-mini-value">${escapeHtml(settings.classifierModel || 'Auto - use Utility')}</div>
      </div>
    </div>
    <div class="mm-overview-actions">
      <button class="btn btn-sm" data-mm-open-pane="advanced" data-mm-focus="mm-role-utility">Edit roles</button>
    </div>
  `;

  const navBadge = q('mm-nav-activity-badge');
  if (navBadge) {
    navBadge.textContent = activeDownloads.length > 0 ? String(activeDownloads.length) : '';
    navBadge.classList.toggle('hidden', activeDownloads.length === 0);
  }

  if (!activeDownloads.length) {
    activityEl.innerHTML = `
      <div class="mm-overview-empty">
        <div class="mm-overview-empty-title">Nothing is downloading right now</div>
        <div class="mm-overview-empty-copy">When you queue new engine files, they will show up here until they finish.</div>
        <button class="btn btn-sm" data-mm-open-pane="activity">Open activity</button>
      </div>
    `;
  } else {
    activityEl.innerHTML = `
      <div class="mm-overview-activity-list">
        ${activeDownloads.slice(0, 3).map((download) => `
          <div class="mm-overview-activity-item">
            <div class="mm-overview-activity-name">${escapeHtml(download.filename || download.repo_id || 'Download')}</div>
            <div class="mm-overview-activity-meta">${escapeHtml(download.stage || 'Preparing')}</div>
          </div>
        `).join('')}
      </div>
      <div class="mm-overview-actions">
        <button class="btn btn-sm" data-mm-open-pane="activity">Open activity</button>
      </div>
    `;
  }

  const featuredNames = engineEntries
    .slice()
    .sort((left, right) => {
      const leftLoaded = left.engineLoaded ? 1 : 0;
      const rightLoaded = right.engineLoaded ? 1 : 0;
      if (leftLoaded !== rightLoaded) return rightLoaded - leftLoaded;
      const leftSelected = left.isSelected ? 1 : 0;
      const rightSelected = right.isSelected ? 1 : 0;
      if (leftSelected !== rightSelected) return rightSelected - leftSelected;
      return Number(right.model?.size || 0) - Number(left.model?.size || 0);
    })
    .slice(0, 3)
    .map((entry) => formatEngineModelRef(entry.model?.name || ''))
    .filter(Boolean);
  libraryEl.innerHTML = `
    <div class="mm-overview-mini-grid">
      <div class="mm-overview-mini-card">
        <div class="mm-overview-mini-label">Engine library</div>
        <div class="mm-overview-mini-value">${escapeHtml(formatCount(engineEntries.length, 'model'))}</div>
      </div>
      <div class="mm-overview-mini-card">
        <div class="mm-overview-mini-label">Storage used</div>
        <div class="mm-overview-mini-value">${escapeHtml(formatBytes(engineStorage))}</div>
      </div>
    </div>
    <div class="mm-overview-chip-row">
      ${featuredNames.length
        ? featuredNames.map((name) => `<span class="mm-overview-chip">${escapeHtml(name)}</span>`).join('')
        : '<span class="mm-overview-chip muted">No engine models installed yet</span>'}
    </div>
    <div class="mm-overview-actions">
      <button class="btn btn-sm" data-mm-open-pane="library">Open library</button>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Modal Creation (lazy)
// ---------------------------------------------------------------------------

function createModal() {
  if (modalEl) return;

  modalEl = document.createElement('div');
  modalEl.className = 'modal-overlay hidden';
  modalEl.id = 'model-manager-modal';
  modalEl.innerHTML = `
    <div class="modal mm-modal-size">
      <div class="modal-header">
        <span class="modal-title">Devices & Models</span>
        <button class="icon-btn small" id="mm-close-btn" title="Close">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>

      <div class="modal-body mm-body">
        <div class="mm-workspace">
          <aside class="mm-sidebar">
            <div class="mm-sidebar-eyebrow">Workspace</div>
            <div class="mm-sidebar-title">Model manager</div>
            <div class="mm-nav">
              <button class="mm-nav-btn active" data-mm-nav="overview" aria-selected="true">
                <span class="mm-nav-label">Overview</span>
                <span class="mm-nav-copy">Status, recommendations, and quick actions</span>
              </button>
              <button class="mm-nav-btn" data-mm-nav="discover" aria-selected="false">
                <span class="mm-nav-label">Discover</span>
                <span class="mm-nav-copy">Find and acquire new local models</span>
              </button>
              <button class="mm-nav-btn" data-mm-nav="library" aria-selected="false">
                <span class="mm-nav-label">Library</span>
                <span class="mm-nav-copy">Manage installed models across devices</span>
              </button>
              <button class="mm-nav-btn" data-mm-nav="activity" aria-selected="false">
                <span class="mm-nav-label">Activity</span>
                <span class="mm-nav-copy">Downloads, retries, and job history</span>
                <span class="mm-nav-badge hidden" id="mm-nav-activity-badge"></span>
              </button>
              <button class="mm-nav-btn" data-mm-nav="advanced" id="mm-nav-advanced" aria-selected="false">
                <span class="mm-nav-label">Advanced</span>
                <span class="mm-nav-copy">Directories, slots, adapters, and controls</span>
              </button>
              <button class="mm-nav-btn" data-mm-nav="fabric" id="mm-nav-fabric" aria-selected="false">
                <span class="mm-nav-label">Fabric</span>
                <span class="mm-nav-copy">Pair other Augmentum instances + share capabilities</span>
              </button>
            </div>
          </aside>

          <div class="mm-main">
            <div class="mm-load-sheet hidden" id="mm-load-sheet">
              <div class="mm-load-sheet-panel">
                <div class="mm-load-sheet-header">
                  <div>
                    <div class="mm-load-sheet-eyebrow" id="mm-load-sheet-eyebrow">Built-in Engine Load Setup</div>
                    <div class="mm-load-sheet-title" id="mm-load-sheet-title">Model</div>
                    <div class="mm-load-sheet-copy" id="mm-load-sheet-copy">Choose how this model should load in memory. Saved defaults are reused whenever you pick this model from the header.</div>
                  </div>
                  <button class="icon-btn small" id="mm-load-sheet-close" title="Close">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                  </button>
                </div>
                <div class="mm-load-sheet-body">
                  <div class="mm-load-sheet-grid">
                    <label class="mm-load-field">
                      <span class="mm-load-label">Context length</span>
                      <input class="field-input" type="number" min="2048" step="1024" id="mm-load-ctx-size">
                      <span class="mm-load-hint" id="mm-load-ctx-hint">Model max: --</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">GPU placement</span>
                      <select class="field-select" id="mm-load-gpu-mode">
                        <option value="auto">Auto (Recommended)</option>
                        <option value="custom">Choose GPU layers</option>
                        <option value="moe_auto_vram" data-moe-only="true">MoE: maximize GPU (auto-balance)</option>
                        <option value="moe_first_n_cpu" data-moe-only="true">MoE: experts of first N layers on CPU</option>
                        <option value="moe_cpu" data-moe-only="true">MoE: all experts on CPU (minimal VRAM)</option>
                        <option value="cpu">Keep on CPU / system RAM</option>
                      </select>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-gpu-layers-wrap">
                      <span class="mm-load-label">GPU layers</span>
                      <input class="field-input" type="number" min="0" step="1" id="mm-load-gpu-layers">
                      <span class="mm-load-hint" id="mm-load-gpu-layers-hint">Higher values keep more of the model on the GPU.</span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-moe-cpu-layers-wrap">
                      <span class="mm-load-label">Expert-offload layers
                        <button type="button" class="btn btn-sm mm-load-inline-btn" id="mm-load-moe-fit-vram" title="Compute the value that maxes GPU utilisation without spilling into shared memory">Fit to VRAM</button>
                      </span>
                      <input class="field-input" type="number" min="0" step="1" id="mm-load-moe-cpu-layers">
                      <span class="mm-load-hint">Push experts of the first N layers to CPU; keep experts of later layers on GPU. Lower = more on GPU (faster, more VRAM). "Fit to VRAM" auto-picks the lowest N that won't spill.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">K cache</span>
                      <select class="field-select" id="mm-load-kv-cache">
                        <option value="">Auto / default</option>
                        <option value="f16">F16 (highest quality)</option>
                        <option value="q8_0">Q8_0 (balanced)</option>
                        <option value="q4_0">Q4_0 (lowest memory)</option>
                      </select>
                      <span class="mm-load-hint">Quantization for the K side of the KV cache.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">V cache</span>
                      <select class="field-select" id="mm-load-kv-cache-v">
                        <option value="">Same as K</option>
                        <option value="f16">F16 (highest quality)</option>
                        <option value="q8_0">Q8_0 (balanced)</option>
                        <option value="q4_0">Q4_0 (lowest memory)</option>
                      </select>
                      <span class="mm-load-hint">Override the V cache quantization independently. K compresses cleanly; V often benefits from a lighter quant.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Batch size</span>
                      <input class="field-input" type="number" min="32" max="8192" step="32" id="mm-load-batch-size">
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">CPU threads (decode)</span>
                      <input class="field-input" type="number" min="0" max="256" step="1" id="mm-load-cpu-threads">
                      <span class="mm-load-hint">0 = auto (llama-server picks half of available threads). Bump for CPU-resident weights (partial offload, MoE experts on CPU).</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">CPU threads (prefill)</span>
                      <input class="field-input" type="number" min="0" max="256" step="1" id="mm-load-cpu-threads-batch">
                      <span class="mm-load-hint">0 = auto. Separate pool for prompt processing. Higher values help long-prompt TTFT when expert eval is CPU-bound.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Lock model in RAM</span>
                      <select class="field-select" id="mm-load-mlock">
                        <option value="false">Off</option>
                        <option value="true">On (--mlock)</option>
                      </select>
                      <span class="mm-load-hint">Pins resident weights so the OS can't swap them out. Useful for long sessions with CPU-resident layers/experts.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Flash attention</span>
                      <select class="field-select" id="mm-load-flash-attn">
                        <option value="true">On</option>
                        <option value="false">Off</option>
                      </select>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Native primer (trained-native model)</span>
                      <select class="field-select" id="mm-load-native-primer" data-default="false">
                        <option value="false">Off — full mode scaffolding</option>
                        <option value="true">On — serve the bare trained primer</option>
                      </select>
                      <span class="mm-load-hint">Only for models trained on Augmentum's own format (e.g. Alethia). Swaps the full mode system-prompt for the bare trained primer (<code>:C</code> + tools line) at egress — the train==serve path. Helps ONLY on surfaces the model was trained on; verify analyze / narrative / build before trusting them. Applies immediately, per-model, and persists.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Vision (mmproj)</span>
                      <select class="field-select" id="mm-load-vision-mode" data-default="auto">
                        <option value="auto">Use global default</option>
                        <option value="true">On (attach projector)</option>
                        <option value="false">Off (text-only)</option>
                      </select>
                      <span class="mm-load-hint">Per-load override. <strong>Attaching the projector disables KV session restore</strong> for this load — upstream llama.cpp returns 501 on /slots/save+restore whenever <code>--mmproj</code> is loaded. Leave Off for narrative/text chats; turn On only when you need the primary model to actually see images. Captioner-based vision (when configured) works in either mode.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">MTP self-speculation</span>
                      <select class="field-select" id="mm-load-mtp-enabled" data-default="auto">
                        <option value="auto">Use global default</option>
                        <option value="true">On (draft-mtp)</option>
                        <option value="false">Off</option>
                      </select>
                      <span class="mm-load-hint">Per-load override. <strong>Requires a GGUF with built-in MTP heads</strong> — look for the MTP-capable chip on the model card. Wins over the external draft below. Forces single-slot.</span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-mtp-n-max-wrap">
                      <span class="mm-load-label">MTP draft length (n_max)</span>
                      <input class="field-input" type="number" min="1" max="16" step="1" id="mm-load-mtp-n-max" placeholder="2">
                      <span class="mm-load-hint">Speculated tokens per draft step. Default <strong>2</strong> — empirically the fastest on Qwen 3.6 27B (1.49× speedup @ 78% acceptance). Higher values lose wall-clock because acceptance falls faster than parallel-verify gains.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Speculative decoding</span>
                      <select class="field-select" id="mm-load-draft-model">
                        <option value="">Off</option>
                      </select>
                      <span class="mm-load-hint" id="mm-load-draft-hint">Manual for now. Pick a local draft model and llama.cpp will validate the pair when loading. <strong>Ignored when MTP self-speculation is on.</strong></span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-draft-max-wrap">
                      <span class="mm-load-label">Draft tokens ahead</span>
                      <input class="field-input" type="number" min="1" max="32" step="1" id="mm-load-draft-max">
                      <span class="mm-load-hint">Start small while testing a pair. Larger values can improve speedups but are less forgiving.</span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-draft-ctx-size-wrap">
                      <span class="mm-load-label">Draft context</span>
                      <input class="field-input" type="number" min="512" max="32768" step="512" id="mm-load-draft-ctx-size">
                      <span class="mm-load-hint">Tokens of KV cache the draft keeps. Drafts only need a small rolling window — 2048 is a good default.</span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-draft-gpu-layers-wrap">
                      <span class="mm-load-label">Draft GPU layers</span>
                      <input class="field-input" type="number" min="0" max="999" step="1" id="mm-load-draft-gpu-layers">
                      <span class="mm-load-hint">999 = full offload (recommended for small drafts). 0 keeps the draft on CPU.</span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-draft-min-wrap">
                      <span class="mm-load-label">Draft min tokens</span>
                      <input class="field-input" type="number" min="0" max="32" step="1" id="mm-load-draft-min">
                      <span class="mm-load-hint">Skip drafting when fewer than this many tokens look likely. 0 disables the gate.</span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-draft-p-min-wrap">
                      <span class="mm-load-label">Draft confidence floor</span>
                      <input class="field-input" type="number" min="0" max="1" step="0.05" id="mm-load-draft-p-min">
                      <span class="mm-load-hint">Stop drafting early when the draft's per-token probability falls below this.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">LoRA adapter</span>
                      <input class="field-input" type="text" id="mm-load-lora-model" placeholder="/models/loras/my-finetune.gguf">
                      <span class="mm-load-hint">Absolute path to a LoRA GGUF/safetensors file. Path validation runs server-side; mismatched architecture will error at startup. Leave blank to disable.</span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-lora-scale-wrap">
                      <span class="mm-load-label">LoRA scale</span>
                      <input class="field-input" type="number" min="0" max="2" step="0.05" id="mm-load-lora-scale">
                      <span class="mm-load-hint">Weight multiplier. 1.0 applies the LoRA at its trained strength; lower softens, higher amplifies (rarely useful past 1.5).</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Sampler seed</span>
                      <input class="field-input" type="number" min="-1" step="1" id="mm-load-seed">
                      <span class="mm-load-hint">-1 = random per request (default). Set a non-negative integer to pin the RNG for reproducible generation.</span>
                    </label>
                    <label class="mm-load-field hidden" data-multi-gpu="true" id="mm-load-tensor-split-wrap">
                      <span class="mm-load-label">Tensor split</span>
                      <input class="field-input" type="text" id="mm-load-tensor-split" placeholder="e.g. 24,16">
                      <span class="mm-load-hint">Comma-separated relative weights for distributing layers across GPUs. "24,16" puts 60% on GPU 0 and 40% on GPU 1.</span>
                    </label>
                    <label class="mm-load-field hidden" data-multi-gpu="true" id="mm-load-main-gpu-wrap">
                      <span class="mm-load-label">Main GPU</span>
                      <input class="field-input" type="number" min="0" max="15" step="1" id="mm-load-main-gpu">
                      <span class="mm-load-hint">Index of the GPU that holds the non-distributable bits (output layer, KV reductions). Usually 0.</span>
                    </label>
                    <label class="mm-load-field hidden" data-multi-gpu="true" id="mm-load-split-mode-wrap">
                      <span class="mm-load-label">Split mode</span>
                      <select class="field-select" id="mm-load-split-mode">
                        <option value="">Auto (layer)</option>
                        <option value="layer">Layer-wise (default)</option>
                        <option value="row">Row-wise</option>
                        <option value="none">None — keep on main GPU</option>
                      </select>
                      <span class="mm-load-hint">How individual tensors get sliced across GPUs. "Layer" distributes whole layers; "row" splits each tensor's rows; "none" forces single-GPU mode.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Unload after idle</span>
                      <input class="field-input" type="number" min="0" step="60" id="mm-load-idle-timeout">
                      <span class="mm-load-hint">Seconds. Set 0 to keep the model loaded until you stop it manually.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Chat template</span>
                      <select class="field-select" id="mm-load-chat-template-mode">
                        <option value="embedded">Embedded (recommended)</option>
                        <option value="builtin">llama-server built-in</option>
                        <option value="custom">Custom (paste Jinja)</option>
                      </select>
                      <span class="mm-load-hint">Embedded uses the GGUF's own chat template (correct for GLM, Qwen3, DeepSeek-R1). Switch to built-in or custom only if reasoning leaks or the model misbehaves.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Reasoning extraction</span>
                      <select class="field-select" id="mm-load-reasoning-format">
                        <option value="">Default (deepseek)</option>
                        <option value="deepseek">DeepSeek (extract &lt;think&gt; to reasoning_content)</option>
                        <option value="none">Inline (preserve raw &lt;think&gt; tags)</option>
                      </select>
                      <span class="mm-load-hint">Where the &lt;think&gt;…&lt;/think&gt; content lands. Default is fine for almost all models.</span>
                    </label>
                    <label class="mm-load-field hidden" id="mm-load-chat-template-content-wrap">
                      <span class="mm-load-label">Custom Jinja template</span>
                      <textarea class="field-input" id="mm-load-chat-template-content" rows="6" spellcheck="false" placeholder="{% for message in messages %}…{% endfor %}"></textarea>
                      <span class="mm-load-hint">Paste a Jinja chat template. Saved to <code>&lt;model_dir&gt;/.chat_templates/&lt;model&gt;.jinja</code> when you load this model.</span>
                    </label>
                    <label class="mm-load-field">
                      <span class="mm-load-label">Template kwargs (JSON)</span>
                      <input class="field-input" id="mm-load-chat-template-kwargs" placeholder='{}'>
                      <span class="mm-load-hint">Esoteric per-model kwargs forwarded to <code>--chat-template-kwargs</code>. Don't use this for thinking on/off — that's the brain icon in the chat composer (auto-detected for Qwen 3.x and GLM-4.x). Use this only for niche template flags that aren't surfaced as a UI control.</span>
                    </label>
                  </div>

                  <div class="mm-load-plan" id="mm-load-plan">
                    <div class="mm-load-plan-card">
                      <span class="mm-load-plan-label">Model max</span>
                      <span class="mm-load-plan-value" id="mm-load-plan-max">--</span>
                    </div>
                    <div class="mm-load-plan-card">
                      <span class="mm-load-plan-label">Estimated load</span>
                      <span class="mm-load-plan-value" id="mm-load-plan-loaded-ctx">--</span>
                    </div>
                    <div class="mm-load-plan-card">
                      <span class="mm-load-plan-label">Peak VRAM</span>
                      <span class="mm-load-plan-value" id="mm-load-plan-vram">--</span>
                    </div>
                    <div class="mm-load-plan-card">
                      <span class="mm-load-plan-label">Peak RAM</span>
                      <span class="mm-load-plan-value" id="mm-load-plan-ram">--</span>
                    </div>
                  </div>

                  <div class="mm-load-plan-note" id="mm-load-plan-note">Adjust settings to see how this model would load.</div>
                  <div class="mm-load-plan-warnings hidden" id="mm-load-plan-warnings"></div>

                  <label class="mm-load-default-toggle" for="mm-load-save-default">
                    <input type="checkbox" id="mm-load-save-default">
                    <span>Save this as the default load setup for this model</span>
                  </label>

                  <div class="mm-load-sheet-actions">
                    <button class="btn btn-sm" id="mm-load-clear-default">Clear saved default</button>
                    <button class="btn btn-sm" id="mm-load-save-default-btn">Save default</button>
                    <button class="btn btn-primary" id="mm-load-apply-btn">Use now</button>
                  </div>
                </div>
              </div>
            </div>

            <section class="mm-pane" data-mm-pane="overview">
              <div class="mm-section">
                <div id="mm-overview-hero"></div>
              </div>

              <div class="mm-section">
                <div class="mm-section-header">
                  <div>
                    <div class="mm-section-title">Recommended for this machine</div>
                    <div class="mm-section-copy">Curated built-in engine picks to get moving quickly.</div>
                  </div>
                  <button class="btn btn-sm" data-mm-select-backend="engine">Discover more</button>
                </div>
                <div id="mm-overview-recommended" class="mm-rec-grid">
                  <div class="mm-empty">Loading recommendations...</div>
                </div>
              </div>

              <div class="mm-overview-grid">
                <div class="mm-section">
                  <div class="mm-section-header">
                    <div>
                      <div class="mm-section-title">Current model</div>
                      <div class="mm-section-copy">The active or next-ready engine model, including its saved load defaults.</div>
                    </div>
                  </div>
                  <div id="mm-overview-current">
                    <div class="mm-empty">Checking the current engine model...</div>
                  </div>
                </div>

                <div class="mm-section">
                  <div class="mm-section-header">
                    <div>
                      <div class="mm-section-title">Task roles</div>
                      <div class="mm-section-copy">Long-term internal assignments for utility and classifier work.</div>
                    </div>
                  </div>
                  <div id="mm-overview-roles">
                    <div class="mm-empty">Loading role assignments...</div>
                  </div>
                </div>
              </div>

              <div class="mm-overview-grid">
                <div class="mm-section">
                  <div class="mm-section-header">
                    <div>
                      <div class="mm-section-title">Active downloads</div>
                      <div class="mm-section-copy">Only in-flight work stays here so the first page stays clean.</div>
                    </div>
                  </div>
                  <div id="mm-overview-activity">
                    <div class="mm-empty">Checking download activity...</div>
                  </div>
                </div>

                <div class="mm-section">
                  <div class="mm-section-header">
                    <div>
                      <div class="mm-section-title">Library snapshot</div>
                      <div class="mm-section-copy">A quick read on your built-in engine library and storage footprint.</div>
                    </div>
                  </div>
                  <div id="mm-overview-library">
                    <div class="mm-empty">Loading library snapshot...</div>
                  </div>
                </div>
              </div>
            </section>

            <section class="mm-pane hidden" data-mm-pane="discover">
              <div class="mm-section">
                <div class="mm-section-header mm-section-header-stack">
                  <div>
                    <div class="mm-section-title">Available devices</div>
                    <div class="mm-section-copy">Choose where new models should live and which runtimes Augmentum can use.</div>
                  </div>
                </div>
                <div class="mm-device-grid" id="mm-device-grid">
                  <div class="mm-empty">Checking your setup...</div>
                </div>
              </div>

              <div class="mm-section mm-find-models">
                <div class="mm-section-header mm-section-header-stack">
                  <div>
                    <div class="mm-section-title">Find models</div>
                    <div class="mm-section-copy" id="mm-backend-help">Pick a device above, then search HuggingFace or paste a model name.</div>
                  </div>
                </div>

                <!-- Hidden state holder: backend selection lives here for JS to
                     read, but the visible selector is the device-card grid above.
                     Clicking a device card calls showDiscoverPane(backend) which
                     updates this select.value. Single source of truth, two-way
                     sync via existing handlers. -->
                <select id="mm-backend-select" hidden aria-hidden="true">
                  <option value="engine">Built-in Engine</option>
                  <option value="ollama">Ollama</option>
                  <option value="llamacpp">llama.cpp</option>
                  <option value="vllm">vLLM Engine</option>
                </select>

                <!-- Active download progress — pinned above the input so users
                     don't have to scroll past chips/results to see status. -->
                <div class="mm-progress-area hidden" id="mm-progress-area">
                  <div class="mm-progress-header">
                    <span class="mm-progress-model" id="mm-progress-model"></span>
                    <button class="mm-cancel-btn" id="mm-cancel-btn" title="Cancel download">Cancel</button>
                  </div>
                  <div class="mm-progress-bar">
                    <div class="mm-progress-fill" id="mm-progress-fill"></div>
                  </div>
                  <div class="mm-progress-status" id="mm-progress-status">Preparing...</div>
                </div>

                <div class="mm-download-row">
                  <div class="mm-search-input-wrap" style="position:relative">
                    <svg class="mm-search-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14">
                      <circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    <input type="text" class="field-input mm-download-input" id="mm-pull-input" placeholder="Search HuggingFace, or paste a model name" style="padding-right:26px">
                    <button type="button" class="mm-pull-clear hidden" id="mm-pull-clear" title="Clear" aria-label="Clear search" style="position:absolute;right:6px;top:50%;transform:translateY(-50%);border:none;background:transparent;color:var(--text-muted);cursor:pointer;font-size:16px;line-height:1;padding:2px 4px">×</button>
                  </div>
                  <button class="btn btn-primary mm-download-btn" id="mm-pull-btn">Download</button>
                </div>

                <!-- Inline target hint + external browse link, sitting close
                     to the input they describe. Moved out of the section header
                     so the "Adding to X" context is right next to the action. -->
                <div class="mm-search-meta">
                  <span class="mm-selection-pill" id="mm-selection-pill">No device selected</span>
                  <span class="mm-target-note" id="mm-target-note"></span>
                  <span class="mm-browse-link" id="mm-browse-link">
                    <a href="https://ollama.com/search" target="_blank" rel="noopener">Browse models on ollama.com
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </a>
                  </span>
                </div>

                <div class="mm-search-results hidden" id="mm-search-results"></div>
                <div class="mm-chips" id="mm-chips"></div>

                <div class="mm-gguf-picker hidden" id="mm-gguf-picker">
                  <div class="mm-section-title">Select Quantization</div>
                  <div class="mm-gguf-dest-row">
                    <label for="mm-gguf-dest">Download to:</label>
                    <select id="mm-gguf-dest" class="field-input"></select>
                  </div>
                  <div id="mm-gguf-dest-usage" class="mm-gguf-dest-usage" style="font-size:var(--text-xs);color:var(--text-muted);margin-top:4px"></div>
                  <div class="mm-gguf-list" id="mm-gguf-list"></div>
                </div>
              </div>
            </section>

            <section class="mm-pane hidden" data-mm-pane="library">
              <div class="mm-section">
                <div class="mm-section-header">
                  <div>
                    <div class="mm-section-title">Library</div>
                    <div class="mm-section-copy">Everything already available across your devices, with engine-first actions close at hand.</div>
                  </div>
                  <input type="text" class="field-input mm-model-search" id="mm-model-search" placeholder="Filter models...">
                </div>
                <div id="mm-model-list" class="mm-model-list">
                  <div class="mm-empty">Loading models...</div>
                </div>
              </div>
            </section>

            <section class="mm-pane hidden" data-mm-pane="activity">
              <div class="mm-section mm-downloads-section" id="mm-downloads-section">
                <div class="mm-section-header">
                  <div>
                    <div class="mm-section-title">Download activity</div>
                    <div class="mm-section-copy">Downloads keep running even if you close this window. Failed or cancelled downloads can be retried in place, cleared from history, or discarded with their partial file.</div>
                  </div>
                  <div style="display:flex;align-items:center;justify-content:flex-end;gap:var(--space-xs);flex-wrap:wrap">
                    <button class="btn btn-sm hidden" id="mm-downloads-clear-history">Clear history</button>
                    <button class="btn btn-sm hidden" id="mm-downloads-discard-partials">Discard saved partials</button>
                    <span id="mm-downloads-count" class="mm-backend-badge" style="font-size:var(--text-xs)"></span>
                  </div>
                </div>
                <div id="mm-downloads-list" class="mm-downloads-list" style="margin-top:var(--space-xs);display:flex;flex-direction:column;gap:var(--space-xs)">
                  <div class="mm-empty">Loading download activity...</div>
                </div>
              </div>
            </section>

            <section class="mm-pane hidden" data-mm-pane="advanced">
              <div class="mm-section">
                <div class="mm-section-header mm-section-header-stack">
                  <div>
                    <div class="mm-section-title">Advanced controls</div>
                    <div class="mm-section-copy">Directories, slots, adapters, and other power-user tools stay here so the main workspace can stay focused.</div>
                  </div>
                </div>
              </div>

              <div class="mm-section">
                <div class="mm-section-header mm-section-header-stack">
                  <div>
                    <div class="mm-section-title">Task roles</div>
                    <div class="mm-section-copy">Long-term model choices for internal work. Both default to Auto — override only if you want a smaller/faster model for tools and routing. Also editable in Settings &rarr; Model.</div>
                  </div>
                </div>
                <div class="mm-role-grid">
                  <label class="mm-role-card" for="mm-role-utility">
                    <span class="mm-role-label">Utility</span>
                    <span class="mm-role-copy">Used for background helpers, tools, and smaller internal tasks.</span>
                    <select class="field-select mm-role-select" id="mm-role-utility">
                      <option value="">Auto - use Primary</option>
                    </select>
                  </label>
                  <label class="mm-role-card" for="mm-role-classifier">
                    <span class="mm-role-label">Classifier</span>
                    <span class="mm-role-copy">Used for routing and quick internal decisions.</span>
                    <select class="field-select mm-role-select" id="mm-role-classifier">
                      <option value="">Auto - use Utility</option>
                    </select>
                  </label>
                  <label class="mm-role-card" for="mm-role-heavyweight">
                    <span class="mm-role-label">Heavyweight</span>
                    <span class="mm-role-copy">Frontier-tier model for quality-critical work — Bug Finder verifier, coder stagnation escalation, /second-opinion. Per-coder-workspace HVY button overrides this default.</span>
                    <select class="field-select mm-role-select" id="mm-role-heavyweight">
                      <option value="">Not configured</option>
                    </select>
                    <label class="mm-role-verify" for="mm-verify-enabled" style="display:flex;align-items:center;gap:8px;margin-top:8px;font-size:12px;color:var(--text-secondary,#9aa0a6);cursor:pointer">
                      <input type="checkbox" id="mm-verify-enabled">
                      <span>Independently verify completed background coder runs — a different model reviews the diff against your ask before you're told it's done. Only runs when a Heavyweight model is set above.</span>
                    </label>
                  </label>
                  <label class="mm-role-card" for="mm-game-planner">
                    <span class="mm-role-label">Game agent — planner</span>
                    <span class="mm-role-copy">Full planning turns when the agent plays a game. Reads frames and returns a strict-JSON plan.</span>
                    <select class="field-select mm-role-select" id="mm-game-planner">
                      <option value="">Auto - use Primary</option>
                    </select>
                  </label>
                  <label class="mm-role-card" for="mm-game-fast">
                    <span class="mm-role-label">Game agent — fast lane</span>
                    <span class="mm-role-copy">Sub-second reflex turns and the scene narrator. Wants a small vision-capable model.</span>
                    <select class="field-select mm-role-select" id="mm-game-fast">
                      <option value="">Auto - use Classifier</option>
                    </select>
                  </label>
                  <label class="mm-role-card" for="mm-visual-verify">
                    <span class="mm-role-label">Visual verification</span>
                    <span class="mm-role-copy">Judges Blender renders and game frames in the coder game-foundry loop. Wants a vision-capable model. Auto uses current vision routing.</span>
                    <select class="field-select mm-role-select" id="mm-visual-verify">
                      <option value="">Auto - current routing</option>
                    </select>
                  </label>
                </div>
              </div>

              <details class="mm-advanced hidden" id="mm-advanced-controls">
                <summary>
                  <span>Advanced controls</span>
                  <span class="mm-advanced-copy">Directories, slots, adapters, and other power-user tools.</span>
                </summary>
                <div class="mm-advanced-body">
                  <div class="mm-engine-dashboard hidden" id="mm-engine-dashboard">
                    <div class="mm-lcpp-header">
                      <div class="mm-lcpp-title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10"/></svg>
                        Built-in Engine
                      </div>
                      <div class="mm-server-led" id="mm-engine-led">
                        <span class="mm-led-dot" id="mm-engine-led-dot"></span>
                        <span id="mm-engine-led-label">...</span>
                      </div>
                    </div>
                    <div id="mm-engine-info" class="mm-advanced-note"></div>
                    <div class="mm-section mm-advanced-section">
                      <div class="mm-section-title">Model Directories</div>
                      <div class="mm-section-copy">Folders Augmentum scans for GGUF files. Add as many as you like — files appear in the model picker once detected.</div>

                      <div id="mm-dirs-platform-banner" class="mm-platform-banner hidden"></div>
                      <div id="mm-dirs-restart-banner" class="mm-restart-banner hidden"></div>

                      <div class="mm-dir-group" id="mm-dirs-active-group">
                        <div class="mm-dir-group-label">Active</div>
                        <div id="mm-engine-dirs" class="mm-dir-list"></div>
                      </div>

                      <div class="mm-dir-group hidden" id="mm-dirs-pending-group">
                        <div class="mm-dir-group-label">Pending — needs restart</div>
                        <div id="mm-engine-host-mounts-list" class="mm-dir-list"></div>
                      </div>

                      <div class="mm-inline-actions">
                        <input type="text" class="field-input" id="mm-engine-add-dir" placeholder="/path/to/your/models">
                        <button class="btn btn-sm" id="mm-engine-browse-btn" title="Browse container filesystem">Browse</button>
                        <button class="btn btn-sm btn-primary" id="mm-engine-add-dir-btn">Add folder</button>
                      </div>
                      <div id="mm-engine-mount-instructions" class="hidden mm-inline-note"></div>
                    </div>

                    <div class="mm-file-browser hidden" id="mm-file-browser">
                      <div id="mm-fb-breadcrumbs" class="mm-file-browser-breadcrumbs"></div>
                      <div id="mm-fb-contents" class="mm-file-browser-contents"></div>
                    </div>
                  </div>

                  <div class="mm-lcpp-dashboard" id="mm-lcpp-dashboard">
                    <div class="mm-lcpp-header">
                      <div class="mm-lcpp-title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
                        llama.cpp Server
                      </div>
                      <div class="mm-server-led" id="mm-server-led">
                        <span class="mm-led-dot" id="mm-led-dot"></span>
                        <span id="mm-led-label">...</span>
                      </div>
                    </div>

                    <div class="mm-lcpp-stats" id="mm-lcpp-stats"></div>

                    <div class="mm-slots-section" id="mm-slots-section">
                      <div class="mm-section-title">Inference Slots</div>
                      <div class="mm-slots-row" id="mm-slots-row"></div>
                    </div>

                    <div class="mm-lora-section hidden" id="mm-lora-section">
                      <div class="mm-section-title">LoRA Adapters</div>
                      <div class="mm-lora-list" id="mm-lora-list"></div>
                      <div class="mm-lora-apply">
                        <button class="btn btn-sm" id="mm-lora-apply-btn">Apply Scales</button>
                      </div>
                    </div>

                    <div class="mm-section mm-advanced-section mm-kv-section">
                      <div class="mm-section-title">KV Cache Persistence</div>
                      <div class="mm-section-copy">How long saved chat states live on disk and how many to keep per model.</div>
                      <div class="mm-kv-grid">
                        <label class="mm-kv-field">
                          <div class="mm-kv-label">Default cache lifetime
                            <span class="mm-kv-unit">days</span>
                            <span class="mm-status-pill mm-status-live" title="Applies immediately, no reload needed">live</span>
                          </div>
                          <input type="number" min="0" max="365" step="1" id="mm-kv-ttl-days" class="field-input" data-kv-key="engine_kv_ttl_days" data-default="2">
                          <div class="mm-kv-help">Sliding window — active chats never expire. <strong>0 = never expire.</strong> Default: 2 days.</div>
                        </label>

                        <label class="mm-kv-field">
                          <div class="mm-kv-label">Narrative cache lifetime
                            <span class="mm-kv-unit">days</span>
                            <span class="mm-status-pill mm-status-live" title="Applies immediately, no reload needed">live</span>
                          </div>
                          <input type="number" min="0" max="365" step="1" id="mm-kv-narrative-ttl-days" class="field-input" data-kv-key="engine_kv_narrative_ttl_days" data-default="7">
                          <div class="mm-kv-help">Same as above but only for narrative chats — they tend to run longer. Default: 7 days.</div>
                        </label>

                        <label class="mm-kv-field">
                          <div class="mm-kv-label">Max snapshots per model
                            <span class="mm-kv-unit">files</span>
                            <span class="mm-status-pill mm-status-live" title="Applies immediately, no reload needed">live</span>
                          </div>
                          <input type="number" min="1" max="100" step="1" id="mm-kv-max-snapshots" class="field-input" data-kv-key="engine_kv_max_snapshots_per_model" data-default="8">
                          <div class="mm-kv-help">Oldest unpinned chats evicted when over this. ~200-500 MB each. Default: 8.</div>
                        </label>

                        <label class="mm-kv-field mm-kv-field-toggle">
                          <input type="checkbox" id="mm-kv-auto-pin-narrative" data-kv-key="engine_kv_auto_pin_narrative" data-default="false">
                          <div>
                            <div class="mm-kv-label">Auto-pin narrative chats
                              <span class="mm-status-pill mm-status-live" title="Applies immediately, no reload needed">live</span>
                            </div>
                            <div class="mm-kv-help">Protects long-running RP from eviction — narrative caches never expire. Default: off.</div>
                          </div>
                        </label>
                      </div>
                      <div class="mm-kv-status" id="mm-kv-status"></div>
                    </div>
                  </div>

                  <!-- Multi-slot KV Routing — sibling of mm-engine-dashboard
                       and mm-lcpp-dashboard so it's visible regardless of
                       which dashboard is active. The KV Cache Persistence
                       section above is gated on caps.has_llamacpp; this
                       one shouldn't be, since the user needs to see it
                       to enable multi-slot in the first place (which then
                       activates the rest of the lcpp dashboard). -->
                  <div class="mm-section mm-advanced-section mm-kv-section">
                    <div class="mm-section-title">Multi-slot KV Routing</div>
                    <div class="mm-section-copy">Lets background tasks (memory extraction, dream cycles) run alongside chat instead of queueing behind it. Default on for hardware that supports it.</div>
                    <div class="mm-kv-grid">
                      <label class="mm-kv-field">
                        <div class="mm-kv-label">Multi-slot KV
                          <span class="mm-kv-unit" id="mm-multislot-resolved-hint"></span>
                          <span class="mm-status-pill mm-status-reload" title="Takes effect on next model load">reload</span>
                        </div>
                        <select id="mm-multislot-enabled" data-kv-key="engine_multislot_enabled" class="field-input" data-default="auto">
                          <option value="auto">Auto (recommended)</option>
                          <option value="true">Always on</option>
                          <option value="false">Always off</option>
                        </select>
                        <div class="mm-kv-help"><strong>Auto</strong> follows the system's current recommendation. <strong>Always on/off</strong> stays put even if the recommendation changes. Default: auto.</div>
                      </label>

                      <label class="mm-kv-field">
                        <div class="mm-kv-label">Parallel slots
                          <span class="mm-kv-unit">slots</span>
                          <span class="mm-status-pill mm-status-reload" title="Takes effect on next model load">reload</span>
                        </div>
                        <input type="number" min="0" max="32" step="1" id="mm-multislot-parallel" data-kv-key="engine_parallel_slots" class="field-input" data-default="0">
                        <div class="mm-kv-help"><strong>0 = auto</strong> (picks 4). Increase for households with concurrent users. ~50-150 MB VRAM per slot. Default: 0.</div>
                      </label>

                      <label class="mm-kv-field">
                        <div class="mm-kv-label">Warm-tier RAM cache
                          <span class="mm-kv-unit">MiB</span>
                          <span class="mm-status-pill mm-status-reload" title="Takes effect on next model load">reload</span>
                        </div>
                        <input type="number" min="0" max="65536" step="256" id="mm-multislot-cache-ram" data-kv-key="engine_cache_ram_mib" class="field-input" data-default="0">
                        <div class="mm-kv-help"><strong>0 = auto-size</strong> from system RAM (~25%, capped at 16 GiB). Override only when constrained. Default: 0.</div>
                      </label>
                    </div>
                  </div>

                  <!-- Engine Defaults — engine- and hardware-level
                       knobs that apply to every model. Settings that
                       vary per model (kv_cache_type, reasoning_format,
                       chat_template_mode, ctx_size, gpu_layers, etc.)
                       live in each model's Load Options sheet, not
                       here. -->
                  <div class="mm-section mm-advanced-section mm-kv-section">
                    <div class="mm-section-title">Engine Defaults</div>
                    <div class="mm-section-copy">Engine- and hardware-level knobs that apply to every model. Per-model settings (KV precision, reasoning format, context size, etc.) live in each model's <em>Load Options</em> sheet under Library.</div>
                    <div class="mm-kv-grid">
                      <label class="mm-kv-field">
                        <div class="mm-kv-label">Idle unload timeout
                          <span class="mm-kv-unit">seconds</span>
                          <span class="mm-status-pill mm-status-live" title="Applies within 30s, no reload needed">live</span>
                        </div>
                        <input type="number" min="0" max="86400" step="60" id="mm-engine-idle-timeout" data-kv-key="engine_idle_timeout" class="field-input" data-default="600">
                        <div class="mm-kv-help">Auto-unload the model after this much inactivity to free VRAM. <strong>0 = never unload.</strong> Default: 600 (10 min).</div>
                      </label>

                      <label class="mm-kv-field mm-kv-field-toggle">
                        <input type="checkbox" id="mm-engine-warm-on-start" data-kv-key="engine_kv_warm_on_start" data-default="true">
                        <div>
                          <div class="mm-kv-label">Warm MRU session on model load
                            <span class="mm-status-pill mm-status-reload" title="Takes effect on next model load">reload</span>
                          </div>
                          <div class="mm-kv-help">After load, restore the most-recently-used chat's KV so its first message skips prefill. Default: on.</div>
                        </div>
                      </label>

                      <label class="mm-kv-field mm-kv-field-toggle">
                        <input type="checkbox" id="mm-engine-flash-attn" data-kv-key="engine_flash_attn" data-default="true">
                        <div>
                          <div class="mm-kv-label">Flash Attention (default)
                            <span class="mm-status-pill mm-status-reload" title="Takes effect on next model load">reload</span>
                          </div>
                          <div class="mm-kv-help">On for Ampere+ GPUs (RTX 30/40/50). <strong>Turn off for Pascal cards</strong> (GTX 10xx) — they don't support it and will fail to load. Default: on.</div>
                        </div>
                      </label>

                      <label class="mm-kv-field mm-kv-field-toggle">
                        <input type="checkbox" id="mm-engine-mtp-enabled" data-kv-key="engine_mtp_enabled" data-default="false">
                        <div>
                          <div class="mm-kv-label">MTP self-speculation
                            <span class="mm-status-pill mm-status-reload" title="Takes effect on next model load">reload</span>
                          </div>
                          <div class="mm-kv-help">Use the model's own next-N predict heads as the speculation source (<code>--spec-type draft-mtp</code>). Requires a GGUF with built-in MTP heads (look for the <strong>MTP-capable</strong> chip on the model card — currently Qwen 3.6 27B, DeepSeek V3/V4). Wins over any draft model. Forces single-slot. Default: off.</div>
                        </div>
                      </label>

                      <label class="mm-kv-field">
                        <div class="mm-kv-label">MTP draft length (n_max)
                          <span class="mm-kv-unit">tokens</span>
                          <span class="mm-status-pill mm-status-reload" title="Takes effect on next model load">reload</span>
                        </div>
                        <input type="number" min="1" max="16" step="1" id="mm-engine-mtp-n-max" data-kv-key="engine_mtp_n_max" class="field-input" data-default="2">
                        <div class="mm-kv-help">How many speculated tokens per draft step. Default <strong>2</strong> — empirically the fastest setting on Qwen 3.6 27B (24 GB consumer GPU, ctx=16K, q8 KV): <strong>n=2 → 34.1 tok/s @ 78% accept, n=6 → 28.5 @ 51%, n=12 → 26.1 @ 31%</strong>. Higher n_max loses wall-clock once acceptance falls below ~60%. Watch <code>mtp_accept_rate</code> in engine_perf logs to tune for your content mix.</div>
                      </label>

                      <label class="mm-kv-field">
                        <div class="mm-kv-label">Health-check timeout
                          <span class="mm-kv-unit">seconds</span>
                          <span class="mm-status-pill mm-status-reload" title="Takes effect on next model load">reload</span>
                        </div>
                        <input type="number" min="60" max="1800" step="30" id="mm-engine-health-timeout" data-kv-key="engine_health_timeout" class="field-input" data-default="900">
                        <div class="mm-kv-help">Max wait for the engine to come up after launch. Bump for huge models on slow disks (HDD, SMB, network mounts). Default: 900 (15 min).</div>
                      </label>

                      <label class="mm-kv-field">
                        <div class="mm-kv-label">Reasoning budget cap
                          <span class="mm-kv-unit">tokens</span>
                          <span class="mm-status-pill mm-status-live" title="Applies on next request">live</span>
                        </div>
                        <input type="number" min="0" max="131072" step="1024" id="mm-engine-reasoning-budget" data-kv-key="engine_reasoning_budget" class="field-input" data-default="0">
                        <div class="mm-kv-help">Cap the hidden chain-of-thought per turn for reasoning models that consume the <code>reasoning_budget</code> template kwarg (Nemotron Omni, etc.). <strong>0 = no cap.</strong> Suggested: 16384.</div>
                      </label>

                      <label class="mm-kv-field">
                        <div class="mm-kv-label">Reasoning grace period
                          <span class="mm-kv-unit">tokens</span>
                          <span class="mm-status-pill mm-status-live" title="Applies on next request">live</span>
                        </div>
                        <input type="number" min="0" max="8192" step="256" id="mm-engine-reasoning-grace" data-kv-key="engine_reasoning_grace_period" class="field-input" data-default="0">
                        <div class="mm-kv-help">Extra tokens past the budget for the model to wrap up its current thought cleanly. Only meaningful when the cap is set. Default: 0.</div>
                      </label>
                    </div>
                  </div>
                </div>
              </details>
            </section>

            <section class="mm-pane hidden" data-mm-pane="fabric">
              <div class="mm-section">
                <div class="mm-section-header mm-section-header-stack">
                  <div>
                    <div class="mm-section-title">Fabric</div>
                    <div class="mm-section-copy">
                      Pair other Augmentum instances on your network. Once paired, each
                      peer advertises its capabilities (LLM models, image pipelines, knowledge
                      packs) and the local instance can transparently route requests to peers
                      that have a model the local box doesn't. Default off -- nothing happens
                      until you enable + pair.
                    </div>
                  </div>
                </div>
              </div>
              <div id="mm-fabric-root"></div>
            </section>
          </div>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(modalEl);
  dom = {};
  bindEvents();
}

// ---------------------------------------------------------------------------
// Event Binding
// ---------------------------------------------------------------------------

function bindEvents() {
  // Close
  q('mm-close-btn').addEventListener('click', closeModelManager);
  modalEl.addEventListener('click', (e) => {
    if (e.target === modalEl) {
      closeModelManager();
      return;
    }

    const navButton = e.target.closest('[data-mm-nav]');
    if (navButton) {
      const pane = navButton.dataset.mmNav;
      setActiveManagerPane(pane);
      // The library pane renders from the last refreshModelList; opening it must
      // reflect current state (post-install/-delete, or after the engine's cold
      // GGUF scan warms) rather than a stale/empty snapshot.
      if (pane === 'library') refreshModelList();
      return;
    }

    const paneButton = e.target.closest('[data-mm-open-pane]');
    if (paneButton) {
      setActiveManagerPane(paneButton.dataset.mmOpenPane, {
        expandAdvanced: paneButton.dataset.mmExpandAdvanced === 'true',
        focusId: paneButton.dataset.mmFocus || undefined,
      });
      return;
    }

    const backendButton = e.target.closest('[data-mm-select-backend]');
    if (backendButton) {
      showDiscoverPane(backendButton.dataset.mmSelectBackend || 'engine');
      return;
    }

    const recommendedButton = e.target.closest('[data-mm-engine-chip]');
    if (recommendedButton) {
      pullModel(recommendedButton.dataset.mmEngineChip, 'engine');
      return;
    }

    const refreshButton = e.target.closest('[data-mm-refresh-setup]');
    if (refreshButton) {
      refreshDeviceSetup();
      return;
    }

    // Overview "Storage by drive" row → jump to the model list filtered to
    // that drive.
    const driveJump = e.target.closest('[data-mm-drive-jump]');
    if (driveJump) {
      modalState.driveFilter = driveJump.dataset.mmDriveJump || null;
      setActiveManagerPane('library');
      applyDriveFilter();
    }
  });

  // Pull / Enter — route based on backend and input format
  function handlePull() {
    const backend = q('mm-backend-select').value;
    const val = q('mm-pull-input').value.trim();
    if (!val) return;

    // vLLM: safetensors models are whole-repo, no per-file quant picker. A repo
    // id or a search term both go through the safetensors search (which offers
    // the "Download whole repo" action per result).
    if (backend === 'vllm') {
      searchHuggingFace(val, 'vllm');
      return;
    }

    // If input looks like a HF repo (has "/" but no ":"), show file picker
    // regardless of backend — user needs to pick a specific GGUF file
    if (val.includes('/') && !val.includes(':')) {
      fetchGgufFiles(val, backend);
      return;
    }

    if (backend === 'llamacpp') {
      fetchGgufFiles(val, backend);
      return;
    }

    if (backend === 'engine') {
      // NEVER auto-resolve a bare word to "first match at the default quant".
      // The box is dual-purpose (search OR paste). Only a target that already
      // names an explicit file/quant (`org/repo:file.gguf` or `name:quant`)
      // is a real download intent — anything else is a SEARCH query, so
      // surface the HF results + quant chips and let the user pick. Enter in
      // the search box must never silently start a download.
      if (val.includes(':') || val.includes('/')) {
        pullModel(val, 'engine');
      } else {
        searchHuggingFace(val, 'engine');
      }
      return;
    }

    // Ollama: a bare tag IS the canonical pull identifier (e.g. llama3.1:8b),
    // and there's no HF quant-picker for it — keep the direct pull.
    pullModel(val, undefined);
  }

  q('mm-pull-btn').addEventListener('click', handlePull);
  q('mm-pull-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); handlePull(); }
  });

  // Debounced HuggingFace search for engine/llamacpp backends
  q('mm-pull-input').addEventListener('input', () => {
    clearTimeout(hfSearchTimer);
    const backend = q('mm-backend-select').value;
    const val = q('mm-pull-input').value.trim();
    updatePullClearVisibility();
    if ((backend === 'engine' || backend === 'llamacpp' || backend === 'vllm') && val.length >= 2) {
      hfSearchTimer = setTimeout(() => searchHuggingFace(val, backend), 300);
    } else {
      hideSearchResults();
    }
  });

  // Clear (×) — one click empties the box and returns to the recommended list
  // (results + file picker dismissed, chips restored). Only visible when there
  // is text, so it can't be mis-clicked on an empty input.
  q('mm-pull-clear')?.addEventListener('click', () => {
    clearTimeout(hfSearchTimer);
    const input = q('mm-pull-input');
    input.value = '';
    const picker = q('mm-gguf-picker');
    if (picker) picker.classList.add('hidden');
    hideSearchResults();
    updatePullClearVisibility();
    input.focus();
  });

  // Cancel
  q('mm-cancel-btn').addEventListener('click', () => {
    if (pullAbortController) pullAbortController.abort();
  });

  // Backend switch — hide options that aren't available
  const backendSelect = q('mm-backend-select');
  const caps = getCapabilities();
  for (const opt of backendSelect.options) {
    if (opt.value === 'engine' && !caps.has_engine) opt.disabled = true;
    if (opt.value === 'ollama' && !caps.has_ollama) opt.disabled = true;
    if (opt.value === 'llamacpp' && !caps.has_llamacpp) opt.disabled = true;
    // vLLM stays selectable even without the engine — browse/queue safetensors,
    // serve once the engine is installed.
  }
  // Default to first available: engine > ollama > llamacpp
  if (caps.has_engine) backendSelect.value = 'engine';
  else if (caps.has_ollama) backendSelect.value = 'ollama';
  else if (caps.has_llamacpp) backendSelect.value = 'llamacpp';

  // Fetch engine models for chips
  fetchEngineChips().then(() => updateBackendUI());

  // Docker Model Runner — uses Ollama API for chat but not for model management.
  // Show CLI instructions instead of the pull/delete UI.
  if (caps.is_docker_model_runner) {
    const downloadSection = q('mm-pull-input')?.closest('.mm-section');
    if (downloadSection) {
      const dmrNote = document.createElement('div');
      dmrNote.className = 'mm-dmr-notice';
      dmrNote.innerHTML = `
        <div class="mm-dmr-title">Docker-managed setup</div>
        <div class="mm-dmr-text">This device reads models from Docker Model Runner, so installs and removals happen in Docker first.</div>
        <code class="mm-dmr-cmd">docker model pull ai/qwen2.5-coder</code>
        <code class="mm-dmr-cmd">docker model list</code>
        <code class="mm-dmr-cmd">docker model rm ai/model-name</code>
        <div class="mm-dmr-text" style="margin-top:8px">
          <a href="https://docs.docker.com/ai/model-runner/" target="_blank" rel="noopener">Docker Model Runner docs</a>
        </div>
      `;
      // Replace the direct download tools with DMR instructions.
      const downloadRow = downloadSection.querySelector('.mm-download-row');
      if (downloadRow) downloadRow.style.display = 'none';
      const targetNote = downloadSection.querySelector('.mm-target-note');
      if (targetNote) targetNote.style.display = 'none';
      const chips = downloadSection.querySelector('.mm-chips');
      if (chips) chips.style.display = 'none';
      const browseLink = downloadSection.querySelector('.mm-browse-link');
      if (browseLink) browseLink.style.display = 'none';
      downloadSection.insertBefore(dmrNote, downloadSection.querySelector('.mm-progress-area'));
    }
  }

  backendSelect.addEventListener('change', updateBackendUI);
  q('mm-load-sheet-close')?.addEventListener('click', closeEngineLoadSheet);
  q('mm-load-clear-default')?.addEventListener('click', async () => {
    const modelName = modalState.engineLoadSheet.modelName;
    if (!modelName) return;
    try {
      await deleteEngineModelLoadProfile(modelName);
      q('mm-load-save-default').checked = false;
      q('mm-load-clear-default').disabled = true;
      q('mm-load-save-default-btn').textContent = 'Save default';
      showToast(`Cleared the saved default for ${modelName}`, 'success');
      await refreshModelList();
    } catch {
      showToast('Could not clear the saved default', 'error');
    }
  });
  q('mm-load-save-default-btn')?.addEventListener('click', async () => {
    try {
      await saveEngineLoadDefaultFromSheet();
      await refreshModelList();
    } catch {
      showToast('Could not save the default load setup', 'error');
    }
  });
  q('mm-load-apply-btn')?.addEventListener('click', applyEngineLoadFromSheet);
  q('mm-load-moe-fit-vram')?.addEventListener('click', fitMoeOffloadToVram);
  // Native-primer toggle — writes the global native_primer_models list, adding
  // or removing THIS model. Separate from the load profile (it's a serving-path
  // setting, not a llama-server arg), so it is NOT in the plan-refresh list and
  // applies immediately on change.
  q('mm-load-native-primer')?.addEventListener('change', async (e) => {
    const modelName = modalState.engineLoadSheet.modelName;
    if (!modelName) return;
    const enable = e.target.value === 'true';
    const low = modelName.toLowerCase();
    let list = (modalState.engineLoadSheet.nativePrimerList || []).slice();
    const matched = list.filter((p) => low.includes(p.toLowerCase()));
    if (enable) {
      if (!matched.length) list.push(modelName);  // add exact name if not already covered
    } else {
      list = list.filter((p) => !low.includes(p.toLowerCase()));  // drop everything this model matches
      const broad = matched.filter((p) => p.toLowerCase() !== low);
      if (broad.length) {
        showToast(`Also cleared shared pattern(s): ${broad.join(', ')} — other models matching these are no longer native-served.`, 'info');
      }
    }
    try {
      await putNativePrimerModels(list);
      modalState.engineLoadSheet.nativePrimerList = list;
      showToast(enable
        ? `Native primer ON for ${modelName} — reload the model or start a new turn to apply.`
        : `Native primer OFF for ${modelName}.`, 'success');
    } catch {
      showToast('Could not update the native primer setting', 'error');
      e.target.value = enable ? 'false' : 'true';  // revert the control
    }
  });
  ['mm-load-ctx-size', 'mm-load-gpu-mode', 'mm-load-gpu-layers', 'mm-load-moe-cpu-layers', 'mm-load-kv-cache', 'mm-load-kv-cache-v', 'mm-load-cpu-threads', 'mm-load-cpu-threads-batch', 'mm-load-mlock', 'mm-load-batch-size', 'mm-load-flash-attn', 'mm-load-vision-mode', 'mm-load-mtp-enabled', 'mm-load-mtp-n-max', 'mm-load-draft-model', 'mm-load-draft-max', 'mm-load-draft-ctx-size', 'mm-load-draft-gpu-layers', 'mm-load-draft-min', 'mm-load-draft-p-min', 'mm-load-lora-model', 'mm-load-lora-scale', 'mm-load-seed', 'mm-load-tensor-split', 'mm-load-main-gpu', 'mm-load-split-mode', 'mm-load-idle-timeout', 'mm-load-chat-template-mode', 'mm-load-chat-template-kwargs', 'mm-load-reasoning-format']
    .forEach((id) => {
      q(id)?.addEventListener('input', () => {
        refreshEngineLoadPlan();
      });
      q(id)?.addEventListener('change', () => {
        refreshEngineLoadPlan();
      });
    });
  bindEngineEvents();
  updateBackendUI();
  syncAdvancedControlsVisibility();
  setActiveManagerPane('overview');

  // Model search filter
  let searchTimer;
  q('mm-model-search')?.addEventListener('input', (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      const query = e.target.value.toLowerCase();
      const modelList = q('mm-model-list');
      const cards = modelList?.querySelectorAll('.mm-model-card') || [];
      cards.forEach(card => {
        const name = (card.dataset.name || '').toLowerCase();
        card.style.display = name.includes(query) ? '' : 'none';
      });
      modelList?.querySelectorAll('.mm-model-group').forEach((group) => {
        const hasVisible = Array.from(group.querySelectorAll('.mm-model-card'))
          .some((card) => card.style.display !== 'none');
        group.style.display = hasVisible ? '' : 'none';
      });
    }, 100);
  });
}

// ---------------------------------------------------------------------------
// Open / Close
// ---------------------------------------------------------------------------

export async function openModelManager(initialPane = 'overview') {
  if (window.innerWidth < 768) closeImagePanel();
  createModal();
  q('mm-pull-input').value = '';
  hideProgress();
  q('mm-advanced-controls').open = false;
  renderDeviceGrid();
  renderRoleSettings();
  renderOverview();
  setActiveManagerPane(initialPane);
  syncAdvancedControlsVisibility();
  try {
    await fetchCapabilities();
    updateModelManagerVisibility();
  } catch { /* best effort */ }
  await fetchEngineChips();
  updateBackendUI();
  await refreshModelList();
  await refreshLcppDashboard();
  await loadActiveDownloads();
  modalEl.classList.remove('hidden');

  // Auto-refresh llama.cpp dashboard while open (every 5s)
  if (lcppRefreshTimer) clearInterval(lcppRefreshTimer);
  lcppRefreshTimer = setInterval(refreshLcppDashboard, 5000);
  // Re-poll downloads on a slower cadence to catch jobs enqueued from
  // another tab and to clear out completed/cancelled rows. The per-card
  // SSE streams give us live progress; this is just for membership churn.
  if (downloadsRefreshTimer) clearInterval(downloadsRefreshTimer);
  downloadsRefreshTimer = setInterval(loadActiveDownloads, 8000);
}

function closeModelManager() {
  if (modalEl) modalEl.classList.add('hidden');
  closeEngineLoadSheet();
  if (pullAbortController) {
    pullAbortController.abort();
    pullAbortController = null;
  }
  // An in-flight HF search must not outlive the modal (leak + a late
  // response would write into a hidden pane).
  if (hfSearchController) {
    hfSearchController.abort();
    hfSearchController = null;
  }
  if (hfSearchTimer) {
    clearTimeout(hfSearchTimer);
    hfSearchTimer = null;
  }
  // Drop cached search results so a "Back to results" can't resurface a stale
  // set from a previous session.
  _lastHfSearch = null;
  if (lcppRefreshTimer) {
    clearInterval(lcppRefreshTimer);
    lcppRefreshTimer = null;
  }
  if (downloadsRefreshTimer) {
    clearInterval(downloadsRefreshTimer);
    downloadsRefreshTimer = null;
  }
  stopSafetensorsWatch();
  closeActiveDownloadStreams();
}

function syncAdvancedControlsVisibility() {
  const advanced = q('mm-advanced-controls');
  const advancedNav = q('mm-nav-advanced');
  if (!advanced) return;
  const caps = getCapabilities();
  const showAdvanced = caps.has_engine || caps.has_llamacpp;
  advanced.classList.toggle('hidden', !showAdvanced);
  if (advancedNav) advancedNav.classList.toggle('hidden', !showAdvanced);
  if (!showAdvanced && modalState.activePane === 'advanced') {
    setActiveManagerPane('overview');
  }
}

async function refreshDeviceSetup() {
  try {
    await fetch('/api/backends/discover', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clear_dismissed: false }),
    });
    await fetchCapabilities();
    updateModelManagerVisibility();
    await fetchEngineChips();
    updateBackendUI();
    await refreshModelList();
    renderRoleSettings();
    await refreshLcppDashboard();
    renderOverview();
    showToast('Setup refreshed', 'success');
  } catch {
    showToast('Could not refresh setup', 'error');
  }
}

// ---------------------------------------------------------------------------
// Engine Model Discovery
// ---------------------------------------------------------------------------

async function fetchEngineChips() {
  try {
    const caps = getCapabilities();
    if (!caps.has_engine) return;

    // Get friendly download aliases from the engine catalog.
    const resp = await fetch('/api/engine/catalog');
    if (!resp.ok) return;
    const data = await resp.json();
    const models = data.models || data.data || [];

    engineChips = models.slice(0, 12).map(m => {
      const name = m.name || m.id || m.model || '?';
      const defaultQuant = m.default_quant || 'q4_k_m';
      return {
        label: prettifyEngineCatalogLabel(name),
        model: `${name}:${defaultQuant}`,
      };
    });
  } catch { /* engine not available */ }
}

// ---------------------------------------------------------------------------
// Backend UI Toggle
// ---------------------------------------------------------------------------

function updateBackendUI() {
  const backendSelect = q('mm-backend-select');
  const caps = getCapabilities();
  for (const opt of backendSelect.options) {
    if (opt.value === 'engine') opt.disabled = !caps.has_engine;
    if (opt.value === 'ollama') opt.disabled = !caps.has_ollama;
    if (opt.value === 'llamacpp') opt.disabled = !caps.has_llamacpp;
    // vLLM stays selectable without the engine — browse/queue safetensors now,
    // serve once installed.
  }
  if (backendSelect.selectedOptions[0]?.disabled) {
    if (caps.has_engine) backendSelect.value = 'engine';
    else if (caps.has_ollama) backendSelect.value = 'ollama';
    else if (caps.has_llamacpp) backendSelect.value = 'llamacpp';
    else if (caps.has_vllm) backendSelect.value = 'vllm';
  }

  const backend = backendSelect.value;
  const isLlamacpp = backend === 'llamacpp';
  const isEngine = backend === 'engine';
  const isVllm = backend === 'vllm';
  const profile = getBackendProfile(backend);

  q('mm-pull-input').placeholder = profile.placeholder;
  q('mm-pull-btn').textContent = isLlamacpp ? 'Choose File' : 'Download';
  q('mm-backend-help').textContent = profile.help;
  q('mm-selection-pill').textContent = `Adding to ${profile.label}`;
  q('mm-target-note').textContent = profile.selectionNote;

  // Chips — show an "installed" check on rows the user already has, so a
  // user doesn't accidentally start a duplicate download. Match logic
  // is best-effort substring (chip 'qwen3.6-27b:ud_q4_k_xl' matches an
  // installed name containing 'qwen3.6-27b'). Cross-backend matches are
  // intentional: if you have qwen3.5:7b in Ollama, the engine's qwen3.5
  // chip still flags as installed — same model family, same disk space
  // wasted on a duplicate.
  // vLLM has no preset chips — the safetensors search is the entry point.
  const chips = isVllm ? [] : isEngine ? engineChips : isLlamacpp ? llamacppChips : ollamaChips;
  const chipsEl = q('mm-chips');
  chipsEl.innerHTML = chips
    .map((c) => {
      const installed = isModelInstalled(c.model);
      return `<button class="mm-chip${installed ? ' mm-chip-installed' : ''}" data-model="${escapeHtml(c.model)}"${installed ? ' title="Already in your library"' : ''}>${installed ? '<span class="mm-chip-check" aria-hidden="true">✓</span>' : ''}${escapeHtml(c.label)}</button>`;
    })
    .join('');
  chipsEl.querySelectorAll('.mm-chip').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (isLlamacpp) {
        q('mm-pull-input').value = btn.dataset.model;
        fetchGgufFiles(btn.dataset.model);
      } else {
        pullModel(btn.dataset.model, isEngine ? 'engine' : undefined);
      }
    });
  });

  // Browse link
  const browseEl = q('mm-browse-link');
  browseEl.innerHTML = `<a href="${escapeHtml(profile.browseUrl)}" target="_blank" rel="noopener">${escapeHtml(profile.browseLabel)} <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></a>`;

  // Hide GGUF picker and search results on switch
  q('mm-gguf-picker').classList.add('hidden');
  q('mm-gguf-list').innerHTML = '';
  hideSearchResults();

  // Show/hide engine dashboard
  const engineDash = q('mm-engine-dashboard');
  if (engineDash) {
    if (caps.has_engine) {
      engineDash.classList.remove('hidden');
      refreshEngineDashboard();
    } else {
      engineDash.classList.add('hidden');
    }
  }

  syncAdvancedControlsVisibility();
  renderDeviceGrid();
  renderOverview();
}

async function refreshEngineModelCatalog(force = false) {
  // Only short-circuit when the cache has actual content. If the previous
  // call landed during the engine's initial GGUF scan (the ~30s window
  // after restart while ``discover_gguf_files`` is still warming), the
  // response was an empty list and would get cached forever. Subsequent
  // lookups via ``resolveEngineModelRecord`` would then "Could not find
  // <model>" until the user manually refreshed the page. Treating empty
  // as "not yet ready" lets the next call self-heal once the engine
  // finishes its scan.
  if (modalState.engineModelCatalog?.length && !force) {
    return modalState.engineModelCatalog;
  }
  let resp;
  try {
    resp = await fetch('/api/engine/v2/models');
  } catch (err) {
    throw new Error('Could not load engine model catalog: ' + (err?.message || err));
  }
  if (!resp.ok) throw new Error('Could not load engine model catalog');
  const data = await resp.json();
  modalState.engineModelCatalog = data.models || [];
  // The configured model dirs (e.g. /models/host, /models/spare) — used to
  // label which drive each model lives on and to drive the per-drive filter.
  modalState.engineModelDirs = data.model_dirs || [];
  return modalState.engineModelCatalog;
}

async function resolveEngineModelRecord(modelName) {
  const target = String(modelName || '').trim().toLowerCase();
  if (!target) return null;
  const catalog = await refreshEngineModelCatalog();
  return catalog.find((entry) => {
    const filename = String(entry.filename || '').toLowerCase();
    const stem = filename.endsWith('.gguf') ? filename.slice(0, -5) : filename;
    return stem === target || String(entry.path || '').toLowerCase().endsWith(`${target}.gguf`);
  }) || null;
}

function readEngineLoadForm() {
  const maxCtx = Number.parseInt(q('mm-load-ctx-size')?.max || '', 10) || 0;
  return normalizeEngineProfile({
    ctx_size: q('mm-load-ctx-size')?.value,
    gpu_layers_mode: q('mm-load-gpu-mode')?.value,
    gpu_layers: q('mm-load-gpu-layers')?.value,
    moe_cpu_layers: q('mm-load-moe-cpu-layers')?.value,
    batch_size: q('mm-load-batch-size')?.value,
    kv_cache_type: q('mm-load-kv-cache')?.value,
    kv_cache_type_v: q('mm-load-kv-cache-v')?.value,
    cpu_threads: q('mm-load-cpu-threads')?.value,
    cpu_threads_batch: q('mm-load-cpu-threads-batch')?.value,
    mlock: q('mm-load-mlock')?.value === 'true',
    flash_attn: q('mm-load-flash-attn')?.value !== 'false',
    draft_model: q('mm-load-draft-model')?.value,
    draft_max: q('mm-load-draft-max')?.value,
    draft_ctx_size: q('mm-load-draft-ctx-size')?.value,
    draft_gpu_layers: q('mm-load-draft-gpu-layers')?.value,
    draft_min: q('mm-load-draft-min')?.value,
    draft_p_min: q('mm-load-draft-p-min')?.value,
    // MTP self-speculation per-load. 'auto' = inherit the global
    // engine_mtp_enabled setting (no key sent), so the backend
    // falls back to manager.mtp_enabled.
    mtp_enabled: q('mm-load-mtp-enabled')?.value,
    mtp_n_max: q('mm-load-mtp-n-max')?.value,
    // Vision (mmproj) per-load. 'auto' = inherit engine_auto_pair_mmproj
    // global default (currently False — KV restore preserved). 'true'
    // forces mmproj attach; 'false' suppresses it. Same tri-state shape
    // as MTP; backend normalizes via _extract_engine_load_options.
    vision_mode: q('mm-load-vision-mode')?.value,
    lora_model: q('mm-load-lora-model')?.value,
    lora_scale: q('mm-load-lora-scale')?.value,
    seed: q('mm-load-seed')?.value,
    tensor_split: q('mm-load-tensor-split')?.value,
    main_gpu: q('mm-load-main-gpu')?.value,
    split_mode: q('mm-load-split-mode')?.value,
    idle_timeout: q('mm-load-idle-timeout')?.value,
    chat_template_mode: q('mm-load-chat-template-mode')?.value,
    chat_template_content: q('mm-load-chat-template-content')?.value,
    chat_template_kwargs: q('mm-load-chat-template-kwargs')?.value,
    reasoning_format: q('mm-load-reasoning-format')?.value,
  }, maxCtx);
}

function syncEngineLoadSheetFields() {
  const mode = q('mm-load-gpu-mode')?.value || 'auto';
  const hasDraftModel = Boolean(q('mm-load-draft-model')?.value);
  const hasLora = Boolean(q('mm-load-lora-model')?.value);
  // Show MTP n_max only when the user picked an explicit on/off, not
  // when they're inheriting the global (auto). For 'false' we still
  // show it as a no-op-display so the saved-profile value remains
  // visible (and editable to flip back on).
  const mtpOverride = q('mm-load-mtp-enabled')?.value || 'auto';
  q('mm-load-mtp-n-max-wrap')?.classList.toggle('hidden', mtpOverride === 'auto' || mtpOverride === 'false');
  q('mm-load-gpu-layers-wrap')?.classList.toggle('hidden', mode !== 'custom');
  q('mm-load-moe-cpu-layers-wrap')?.classList.toggle('hidden', mode !== 'moe_first_n_cpu');
  q('mm-load-lora-scale-wrap')?.classList.toggle('hidden', !hasLora);
  q('mm-load-draft-max-wrap')?.classList.toggle('hidden', !hasDraftModel);
  q('mm-load-draft-ctx-size-wrap')?.classList.toggle('hidden', !hasDraftModel);
  q('mm-load-draft-gpu-layers-wrap')?.classList.toggle('hidden', !hasDraftModel);
  q('mm-load-draft-min-wrap')?.classList.toggle('hidden', !hasDraftModel);
  q('mm-load-draft-p-min-wrap')?.classList.toggle('hidden', !hasDraftModel);
  const tmplMode = q('mm-load-chat-template-mode')?.value || 'embedded';
  q('mm-load-chat-template-content-wrap')?.classList.toggle('hidden', tmplMode !== 'custom');
}

async function populateEngineDraftModelOptions(modelName, selectedDraftModel = '') {
  const select = q('mm-load-draft-model');
  const hint = q('mm-load-draft-hint');
  if (!select) return;

  const targetName = formatEngineModelRef(modelName).toLowerCase();
  const selectedCanonical = canonicalEngineModelRef(selectedDraftModel).toLowerCase();
  const selectedStem = formatEngineModelRef(selectedDraftModel).toLowerCase();
  let catalog;
  try {
    catalog = await refreshEngineModelCatalog(true);
  } catch (err) {
    // A failed catalog fetch previously left a silently-empty dropdown —
    // indistinguishable from "you have no other models" (error-vs-empty
    // discipline). Say what actually happened.
    select.innerHTML = '<option value="">Off</option>';
    select.disabled = true;
    if (hint) {
      hint.textContent =
        `Couldn't load the model catalog (${String(err?.message || err).slice(0, 120)}) — `
        + 'close and reopen this sheet to retry.';
    }
    return;
  }
  const options = [...catalog]
    .filter((entry) => formatEngineModelRef(entry.path || entry.filename || '').toLowerCase() !== targetName)
    .sort((left, right) => {
      const leftSize = Number(left.total_size_bytes || left.size || 0);
      const rightSize = Number(right.total_size_bytes || right.size || 0);
      if (leftSize !== rightSize) return leftSize - rightSize;
      return formatEngineModelRef(left.path || left.filename || '').localeCompare(
        formatEngineModelRef(right.path || right.filename || ''),
        undefined,
        { sensitivity: 'base' },
      );
    });

  select.innerHTML = [
    '<option value="">Off</option>',
    ...options.map((entry) => {
      const value = entry.path || '';
      const labelParts = [formatEngineModelRef(value)];
      const sizeBytes = Number(entry.total_size_bytes || entry.size || 0);
      if (sizeBytes > 0) labelParts.push(formatBytes(sizeBytes));
      if (entry.architecture) labelParts.push(String(entry.architecture));
      return `<option value="${escapeHtml(value)}">${escapeHtml(labelParts.join(' Â· '))}</option>`;
    }),
  ].join('');

  const selectedEntry = options.find((entry) => {
    const optionCanonical = canonicalEngineModelRef(entry.path || '').toLowerCase();
    const optionStem = formatEngineModelRef(entry.path || '').toLowerCase();
    return optionCanonical === selectedCanonical || optionStem === selectedStem;
  });
  select.value = selectedEntry?.path || '';
  select.disabled = options.length === 0;
  if (hint) {
    hint.textContent = options.length > 0
      ? 'Manual for now. Pick a local draft model and llama.cpp will validate the pair when loading.'
      : 'Download another local engine model to try speculative decoding.';
  }
}

function renderEngineLoadPlan(plan) {
  if (!plan) return;
  q('mm-load-plan-max').textContent = `${formatTokenCount(plan.profile?.context_length)} tokens`;
  q('mm-load-plan-loaded-ctx').textContent = `${formatTokenCount(plan.applied?.ctx_size)} tokens`;
  q('mm-load-plan-vram').textContent = formatBytes((plan.memory?.estimated_vram_mb || 0) * 1024 * 1024);
  q('mm-load-plan-ram').textContent = formatBytes((plan.memory?.estimated_ram_mb || 0) * 1024 * 1024);
  q('mm-load-ctx-hint').textContent = `Model max: ${formatTokenCount(plan.profile?.context_length)} tokens`;
  // The /plan endpoint scans the GGUF header fresh (and caches the profile),
  // so plan.profile.context_length is authoritative — even for a just-added
  // model whose profile wasn't cached yet when the catalog was fetched. The
  // ctx-size input's ``max`` was seeded from the (possibly missing) catalog
  // record and can be stuck at the 8192 fallback; without this sync,
  // readEngineLoadForm re-clamps every manual ctx entry back down to that
  // stale max. Update it to the real model max so a chosen 131072 sticks.
  const planMaxCtx = Number(plan.profile?.context_length || 0);
  if (planMaxCtx > 0) {
    const ctxInput = q('mm-load-ctx-size');
    if (ctxInput && Number(ctxInput.max || 0) !== planMaxCtx) {
      ctxInput.max = String(planMaxCtx);
    }
  }

  // Multi-GPU fields are noise on single-GPU hosts. The plan response
  // is the source of truth — ``plan.memory.gpu_count`` comes from the
  // nvidia-smi probe at plan time. Hide the section when 0 or 1.
  const gpuCount = Number(plan.memory?.gpu_count || 0);
  document.querySelectorAll('[data-multi-gpu="true"]').forEach((el) => {
    el.classList.toggle('hidden', gpuCount <= 1);
  });

  // MoE expert-offload modes are only meaningful for MoE GGUFs. Hide the
  // <option> elements (and the N-layers wrapper) for dense models so the
  // dropdown isn't littered with no-op choices. The plan response is the
  // authoritative source of is_moe — the /models catalog doesn't carry
  // architecture info, so we discover MoE-ness on first plan refresh.
  const isMoe = Boolean(plan.profile?.is_moe);
  const modeSelect = q('mm-load-gpu-mode');
  if (modeSelect) {
    modeSelect.querySelectorAll('option[data-moe-only="true"]').forEach((opt) => {
      opt.hidden = !isMoe;
      opt.disabled = !isMoe;
    });
    // If a saved profile has a MoE mode but the loaded model is dense
    // (e.g. profile shared between machines, or model was replaced),
    // the backend has already auto-fallen-back to 'auto' in the plan's
    // applied dict. Mirror that on the select.
    if (!isMoe && ['moe_cpu', 'moe_first_n_cpu'].includes(modeSelect.value)) {
      modeSelect.value = plan.applied?.gpu_layers_mode || 'auto';
      syncEngineLoadSheetFields();
    }
  }

  const moeN = plan.applied?.moe_cpu_layers || 0;
  const totalLayers = plan.profile?.n_layers || 0;
  const gpuMode = plan.applied?.gpu_layers_mode === 'cpu'
    ? 'CPU / system RAM'
    : plan.applied?.gpu_layers_mode === 'custom'
      ? `${plan.applied?.gpu_layers || 0} GPU layers`
      : plan.applied?.gpu_layers_mode === 'moe_cpu'
        ? 'MoE: all experts on CPU'
        : plan.applied?.gpu_layers_mode === 'moe_first_n_cpu'
          ? `MoE: experts of first ${moeN}/${totalLayers} on CPU`
          : plan.applied?.gpu_layers_mode === 'moe_auto_vram'
            ? `MoE: VRAM-balanced (N=${moeN}/${totalLayers})`
            : `Auto GPU · ${plan.applied?.gpu_layers || 0} layers`;
  const noteParts = [gpuMode, formatIdleTimeout(plan.applied?.idle_timeout || 0)];
  if (plan.applied?.draft_model) {
    noteParts.push(`Draft ${formatEngineModelRef(plan.applied.draft_model)}`);
  }
  if (plan.memory?.gpu_free_mib) {
    noteParts.push(`${formatBytes(plan.memory.gpu_free_mib * 1024 * 1024)} free on GPU`);
  }
  if (plan.memory?.ram_available_mib) {
    noteParts.push(`${formatBytes(plan.memory.ram_available_mib * 1024 * 1024)} free RAM`);
  }
  if ((plan.memory?.workspace_vram_mb || 0) > 0 || (plan.memory?.workspace_ram_mb || 0) > 0) {
    noteParts.push('includes prompt-processing headroom');
  }
  q('mm-load-plan-note').textContent = noteParts.join(' · ');

  const warningsEl = q('mm-load-plan-warnings');
  const warnings = [...(plan.warnings || [])];
  warningsEl.classList.toggle('hidden', warnings.length === 0);
  warningsEl.innerHTML = warnings.map((warning) => `<div>${escapeHtml(warning)}</div>`).join('');
}

async function fitMoeOffloadToVram() {
  // Ask the backend planner what N (--n-cpu-moe) maximises VRAM use
  // for this model right now, then drop that value into the manual
  // slider and refresh the preview. Keeps the user in "experts of
  // first N layers" mode (so they can still tweak afterwards) but
  // gives them a one-click starting point that's better than guessing.
  const modelName = modalState.engineLoadSheet.modelName;
  if (!modelName) return;
  const fitBtn = q('mm-load-moe-fit-vram');
  if (fitBtn) {
    fitBtn.disabled = true;
    fitBtn.textContent = 'Computing…';
  }
  try {
    const form = readEngineLoadForm();
    // Force moe_auto_vram for this one call so the planner runs the
    // autofit; the user's saved mode is preserved in ``form``.
    const probe = { ...form, gpu_layers_mode: 'moe_auto_vram' };
    const resp = await fetch('/api/engine/v2/models/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelName, ...probe }),
    });
    if (!resp.ok) throw new Error('Could not compute VRAM-balanced N');
    const plan = await resp.json();
    const n = Number(plan?.applied?.moe_cpu_layers);
    if (!Number.isFinite(n) || n < 0) throw new Error('Planner returned no value');
    const input = q('mm-load-moe-cpu-layers');
    if (input) {
      input.value = String(n);
      // Stay in moe_first_n_cpu — keeping the slider mode means the
      // user can nudge N up/down from this autofit baseline without
      // losing control to the auto-recomputer.
      const mode = q('mm-load-gpu-mode');
      if (mode && mode.value !== 'moe_first_n_cpu') mode.value = 'moe_first_n_cpu';
    }
    await refreshEngineLoadPlan();
    showToast(`Set N = ${n} (max-VRAM fit)`, 'success');
  } catch (err) {
    showToast(err?.message || 'Could not compute VRAM-balanced N', 'error');
  } finally {
    if (fitBtn) {
      fitBtn.disabled = false;
      fitBtn.textContent = 'Fit to VRAM';
    }
  }
}

async function refreshEngineLoadPlan() {
  if (!modalState.engineLoadSheet.modelName) return;
  syncEngineLoadSheetFields();
  const payload = {
    model: modalState.engineLoadSheet.modelName,
    ...readEngineLoadForm(),
  };
  try {
    const resp = await fetch('/api/engine/v2/models/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error('Could not preview load settings');
    const plan = await resp.json();
    renderEngineLoadPlan(plan);
  } catch (err) {
    q('mm-load-plan-note').textContent = err.message || 'Could not preview load settings';
  }
}

function closeEngineLoadSheet() {
  q('mm-load-sheet')?.classList.add('hidden');
  modalState.engineLoadSheet = {
    modelName: '', modelPath: '', source: 'manager', slot: 'A',
  };
}

// Native-primer serving toggle (per-model). Backed by the global
// ``native_primer_models`` setting — a comma-separated list of model-name
// substrings. The Load Setup toggle manages this model's membership in that
// list, so a natively-trained model (Alethia) can be opted into the bare
// trained-primer serving path from the same place you set ctx/GPU.
async function fetchNativePrimerModels() {
  try {
    const resp = await fetch('/api/config/tools', { credentials: 'same-origin' });
    if (!resp.ok) return [];
    const data = await resp.json();
    return String((data && data.native_primer_models) || '')
      .split(',').map((s) => s.trim()).filter(Boolean);
  } catch {
    return [];
  }
}

async function putNativePrimerModels(list) {
  const resp = await fetch('/api/config/tools', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({ native_primer_models: list.join(',') }),
  });
  if (!resp.ok) throw new Error(`native_primer save failed (${resp.status})`);
}

export async function openEngineLoadSheet(modelName, opts = {}) {
  if (!modalEl || modalEl.classList.contains('hidden')) {
    await openModelManager();
  } else if (!modalState.inventory) {
    await refreshModelList();
  }
  const record = await resolveEngineModelRecord(modelName);
  if (!record) {
    showToast(`Could not find ${modelName} in the Built-in Engine library`, 'error');
    return false;
  }

  await waitForUiSettingsReady();
  const savedProfile = getEngineModelLoadProfile(modelName);
  // Fall back to the backend's last-load row when there's no explicit
  // Save-Default profile. The /api/engine/v2/models/load route auto-
  // persists every successful load to ``engine.last_load.<id>``, so
  // this gives the form a sensible seed (including per-load MTP +
  // any knob the frontend's engineModelLoadProfiles map happens to
  // omit) without forcing the user to click Save Default.
  let lastLoadProfile = null;
  if (!savedProfile) {
    try {
      const resp = await fetch(`/api/engine/v2/models/last-load?model=${encodeURIComponent(modelName)}`);
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.load_options && Object.keys(data.load_options).length) {
          lastLoadProfile = data.load_options;
        }
      }
    } catch {
      // Best-effort — sheet still opens with empty defaults on failure.
    }
  }
  const seed = savedProfile || lastLoadProfile || {};
  const profile = normalizeEngineProfile(seed, record.context_length || 0);

  const slot = ENGINE_SLOTS[opts.slot] ? opts.slot : 'A';
  modalState.engineLoadSheet = {
    modelName,
    modelPath: record.path || '',
    source: opts.source || 'manager',
    slot,
  };

  q('mm-load-sheet').classList.remove('hidden');
  // Name the destination slot in the sheet itself. The same form serves all
  // three slots now, so without this the user has no way to tell which one a
  // load is about to occupy — and B/C loads are the ones with consequences
  // (B holds VRAM resident; C swaps the model behind the classifier, utility
  // and vision roles).
  q('mm-load-sheet-title').textContent = slot === 'A'
    ? modelName
    : `${modelName} → ${ENGINE_SLOTS[slot].label}`;
  const eyebrowEl = q('mm-load-sheet-eyebrow');
  if (eyebrowEl) eyebrowEl.textContent = ENGINE_SLOTS[slot].eyebrow;
  const copyEl = q('mm-load-sheet-copy');
  if (copyEl) copyEl.textContent = ENGINE_SLOTS[slot].copy;
  // Native-primer toggle reflects whether THIS model is covered by the global
  // native_primer_models list (any entry that is a substring of the name).
  try {
    const npList = await fetchNativePrimerModels();
    modalState.engineLoadSheet.nativePrimerList = npList;
    const covered = npList.some((p) => modelName.toLowerCase().includes(p.toLowerCase()));
    const npSel = q('mm-load-native-primer');
    if (npSel) npSel.value = covered ? 'true' : 'false';
  } catch { /* best effort — leave at default Off */ }
  q('mm-load-ctx-size').value = String(profile.ctx_size);
  q('mm-load-ctx-size').max = String(record.context_length || profile.ctx_size);
  q('mm-load-gpu-mode').value = profile.gpu_layers_mode;
  q('mm-load-gpu-layers').value = String(profile.gpu_layers || 0);
  // Show blank when unset (let the backend autofit decide). Use ``??``
  // so an explicit ``0`` (all experts on GPU, valid for tiny MoEs) is
  // still rendered as "0" rather than collapsing to blank. The prior
  // ``|| 16`` populated 16 for any blank field, which got persisted
  // back on submit and caused OOMs on big MoEs — see the matching
  // normalizeEngineProfile change.
  q('mm-load-moe-cpu-layers').value = profile.moe_cpu_layers == null
    ? ''
    : String(profile.moe_cpu_layers);
  q('mm-load-kv-cache').value = profile.kv_cache_type || '';
  q('mm-load-kv-cache-v').value = profile.kv_cache_type_v || '';
  q('mm-load-cpu-threads').value = String(profile.cpu_threads || 0);
  q('mm-load-cpu-threads-batch').value = String(profile.cpu_threads_batch || 0);
  q('mm-load-mlock').value = profile.mlock ? 'true' : 'false';
  q('mm-load-seed').value = String(profile.seed ?? -1);
  q('mm-load-tensor-split').value = profile.tensor_split || '';
  q('mm-load-main-gpu').value = String(profile.main_gpu || 0);
  q('mm-load-split-mode').value = profile.split_mode || '';
  q('mm-load-lora-model').value = profile.lora_model || '';
  q('mm-load-lora-scale').value = String(profile.lora_scale ?? 1.0);
  q('mm-load-batch-size').value = String(profile.batch_size);
  q('mm-load-flash-attn').value = profile.flash_attn ? 'true' : 'false';
  await populateEngineDraftModelOptions(modelName, profile.draft_model || '');
  q('mm-load-draft-max').value = String(profile.draft_max || 5);
  q('mm-load-draft-ctx-size').value = String(profile.draft_ctx_size || 2048);
  q('mm-load-draft-gpu-layers').value = String(profile.draft_gpu_layers ?? 999);
  q('mm-load-draft-min').value = String(profile.draft_min ?? 1);
  q('mm-load-draft-p-min').value = String(profile.draft_p_min ?? 0.75);
  // MTP self-speculation per-load. Tri-state: explicit true/false or
  // 'auto' (inherit the engine_mtp_enabled global default). Render
  // 'auto' for any saved profile that doesn't pin the field.
  const mtpVal = profile.mtp_enabled;
  q('mm-load-mtp-enabled').value = (mtpVal === true) ? 'true'
    : (mtpVal === false) ? 'false'
    : 'auto';
  q('mm-load-mtp-n-max').value = profile.mtp_n_max != null ? String(profile.mtp_n_max) : '';
  // Vision (mmproj) per-load. Same tri-state shape as MTP.
  const visionVal = profile.vision_mode;
  q('mm-load-vision-mode').value = (visionVal === true) ? 'true'
    : (visionVal === false) ? 'false'
    : 'auto';
  q('mm-load-idle-timeout').value = String(profile.idle_timeout);
  q('mm-load-chat-template-mode').value = profile.chat_template_mode || 'embedded';
  q('mm-load-chat-template-content').value = profile.chat_template_content || '';
  q('mm-load-chat-template-kwargs').value = profile.chat_template_kwargs || '';
  q('mm-load-reasoning-format').value = profile.reasoning_format || '';
  q('mm-load-save-default').checked = Boolean(savedProfile);
  q('mm-load-clear-default').disabled = !savedProfile;
  q('mm-load-save-default-btn').textContent = savedProfile ? 'Update default' : 'Save default';

  await refreshEngineLoadPlan();
  return true;
}

async function saveEngineLoadDefaultFromSheet() {
  const modelName = modalState.engineLoadSheet.modelName;
  if (!modelName) return;
  const profile = readEngineLoadForm();
  await saveEngineModelLoadProfile(modelName, profile);
  q('mm-load-save-default').checked = true;
  q('mm-load-clear-default').disabled = false;
  q('mm-load-save-default-btn').textContent = 'Update default';
  showToast(`Saved default load setup for ${modelName}`, 'success');
}

async function applyEngineLoadFromSheet() {
  const modelName = modalState.engineLoadSheet.modelName;
  if (!modelName) return;
  const source = modalState.engineLoadSheet.source;
  const applyBtn = q('mm-load-apply-btn');
  applyBtn.disabled = true;
  applyBtn.textContent = 'Loading...';
  try {
    const profile = readEngineLoadForm();
    if (q('mm-load-save-default').checked) {
      await saveEngineModelLoadProfile(modelName, profile);
      q('mm-load-clear-default').disabled = false;
      q('mm-load-save-default-btn').textContent = 'Update default';
    }
    const slot = ENGINE_SLOTS[modalState.engineLoadSheet.slot] ? modalState.engineLoadSheet.slot : 'A';
    const spec = ENGINE_SLOTS[slot];
    const body = { [spec.modelField]: modelName, ...profile };
    // Slot C is RESIDENT by contract — the voice/architect routers run on a
    // ~2.5s budget and a cold reload blows it. The server defaults this too;
    // sending it explicitly keeps the sheet honest about what it just did.
    if (slot === 'C') body.idle_timeout = 0;
    let resp = await fetch(spec.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body),
    });
    // Slot C refuses to displace a running external Docker classifier without
    // an explicit confirmation — taking over STOPS that container, which is a
    // visible change to the user's stack, not an implementation detail.
    if (resp.status === 409 && slot === 'C') {
      const data = await resp.json().catch(() => ({}));
      const ok = window.confirm(
        `${data.detail || 'An external classifier container currently serves this role.'}\n\n`
        + 'Take over the classifier role with Slot C? The external container will be stopped.',
      );
      if (!ok) return;
      resp = await fetch(spec.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ ...body, take_over: true }),
      });
    }
    if (resp.status === 507) {
      // Admission control. Report the actual numbers — "insufficient VRAM"
      // with no figures gives the user nothing to act on.
      const data = await resp.json().catch(() => ({}));
      const need = Math.round((data.needed_mb || 0) / 1024 * 10) / 10;
      const free = Math.round((data.free_mb || 0) / 1024 * 10) / 10;
      throw new Error(
        `${data.detail || 'Not enough VRAM'} (needs ~${need} GB, ${free} GB free). `
        + 'Unload something, or re-run with Force.',
      );
    }
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || data.error || 'Load failed');
    }
    if (slot === 'A') {
      await adoptLoadedModel(modelName);
    } else {
      // Point the ROLE at what was just loaded, so the pick is a complete
      // statement of intent rather than "loaded, now go set it in Settings".
      // Safe as of the Slot B/C routing pins: naming the model resolves to
      // the slot that holds it instead of swapping the primary engine.
      await persistSlotRoleModel(slot, modelName);
    }
    await refreshModelList();
    await refreshEngineDashboard();
    showToast(
      slot === 'A' ? `Loaded ${modelName}` : `Loaded ${modelName} into ${ENGINE_SLOTS[slot].label}`,
      'success',
    );
    closeEngineLoadSheet();
    if (source === 'selector') closeModelManager();
  } catch (err) {
    showToast(err.message || 'Load failed', 'error');
  } finally {
    applyBtn.disabled = false;
    applyBtn.textContent = 'Use now';
  }
}

/** Point a slot's ROLE setting at the model just loaded into it.
 *
 * Loading a model into Slot B/C used to be only half the job: the subprocess
 * came up holding the model, but the setting that decides which model the role
 * ASKS for still said something else, so the user had to go find it in
 * Settings. Writing both together is what makes the one-tap pick a complete
 * statement — the same reason the load routes already persist
 * ``engine_secondary_model`` / ``classifier_slot_model`` themselves.
 *
 * Slot C additionally writes ``classifier_model`` (the role's own setting).
 * That is only correct because the classifier slot now PINS its loaded model:
 * before the pin, naming the model here resolved it through the catalog map to
 * the primary engine and made Slot A swap on every classifier call.
 */
async function persistSlotRoleModel(slot, modelName) {
  const key = slot === 'B' ? 'engine_secondary_model' : 'classifier_model';
  try {
    const resp = await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ [key]: modelName }),
    });
    if (!resp.ok) throw new Error(String(resp.status));
  } catch (err) {
    // Loud, not silent: the model IS loaded, but the role still points
    // elsewhere, and that mismatch is exactly the class of bug that cost a
    // fortnight of "why is it using the wrong model".
    showToast(
      `Loaded into ${ENGINE_SLOTS[slot].label}, but could not update the role setting `
      + `(${key}). Set it in Models → Task roles.`,
      'error',
    );
  }
}

/** One-tap entry point for "load this model into slot A/B/C".
 *
 * Honors the saved per-model load profile when there is one (so the second and
 * later loads of a given model are genuinely one tap) and falls back to the
 * load sheet when there isn't — the user picks ctx/GPU/mmproj rather than
 * having them chosen silently, which for Slot C is the difference between
 * having vision and quietly not having it.
 */
export async function loadModelIntoSlot(modelName, slot, { source = 'selector' } = {}) {
  const spec = ENGINE_SLOTS[slot];
  if (!spec) return false;
  await waitForUiSettingsReady();
  const profile = getEngineModelLoadProfile(modelName);
  if (!profile) {
    // First time for this model — let the user set ctx/GPU/mmproj. The sheet
    // lives inside the model manager, so this DOES open it; that is the one
    // screen we deliberately keep, and only until a default is saved.
    await openEngineLoadSheet(modelName, { source, slot });
    return false;
  }

  // Saved profile → straight to the endpoint. Explicitly NOT routed through
  // the sheet: opening it calls openModelManager(), and making the one-tap
  // path pop the full manager modal would rebuild the very "multiple screens"
  // problem this control exists to remove.
  const body = { [spec.modelField]: modelName, ...profile };
  if (slot === 'C') body.idle_timeout = 0;
  // Persistent ('loading' → duration 0) rather than a 3s toast: a cold load
  // runs 5s for a small model and ~140s for a 35B, and a progress indicator
  // that vanishes at 3s reads as "it finished" or "it died".
  const toastId = showToast(`Loading ${modelName} into ${spec.label}…`, 'loading');
  const post = (extra = {}) => fetch(spec.endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({ ...body, ...extra }),
  });
  try {
    let resp = await post();
    if (resp.status === 409 && slot === 'C') {
      const data = await resp.json().catch(() => ({}));
      const ok = window.confirm(
        `${data.detail || 'An external classifier container currently serves this role.'}\n\n`
        + 'Take over the classifier role with Slot C? The external container will be stopped.',
      );
      if (!ok) return false;
      resp = await post({ take_over: true });
    }
    if (resp.status === 507) {
      const data = await resp.json().catch(() => ({}));
      const need = Math.round((data.needed_mb || 0) / 1024 * 10) / 10;
      const free = Math.round((data.free_mb || 0) / 1024 * 10) / 10;
      throw new Error(
        `${data.detail || 'Not enough VRAM'} (needs ~${need} GB, ${free} GB free).`,
      );
    }
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || data.error || 'Load failed');
    }
    if (slot === 'A') {
      await adoptLoadedModel(modelName);
    } else {
      await persistSlotRoleModel(slot, modelName);
    }
    showToast(`Loaded ${modelName} into ${spec.label}`, 'success');
    return true;
  } catch (err) {
    showToast(err.message || `Could not load into ${spec.label}`, 'error');
    return false;
  } finally {
    dismissToast(toastId);
  }
}

async function loadEngineModelFromManager(modelName, { source = 'manager' } = {}) {
  await waitForUiSettingsReady();
  const savedProfile = getEngineModelLoadProfile(modelName);
  if (!savedProfile) {
    await openEngineLoadSheet(modelName, { source });
    return;
  }

  try {
    const resp = await fetch('/api/engine/v2/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelName, ...savedProfile }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || 'Load failed');
    }
    await adoptLoadedModel(modelName);
    await refreshModelList();
    await refreshEngineDashboard();
    showToast(`Loaded ${modelName}`, 'success');
  } catch (err) {
    showToast(err.message || 'Load failed', 'error');
  }
}

// Load a model into the SECOND resident slot ("Slot B"), kept alongside
// the primary so two local models stay loaded at once. Reuses the
// model's saved load profile (gpu-layer cap, ctx, idle timeout) — the
// same per-model config the primary slot uses, so the model lands its
// configured resource footprint whichever slot it's loaded into.
async function loadEngineModelIntoSlotB(modelName, { force = false } = {}) {
  await waitForUiSettingsReady();
  const savedProfile = getEngineModelLoadProfile(modelName) || {};
  try {
    const resp = await fetch('/api/engine/v2/secondary/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelName, ...savedProfile, force }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      // Admission control rejected it (507): the model likely won't fit
      // alongside what's already resident. Offer a force retry rather than
      // dead-ending — the user may want partial CPU offload or knows it
      // fits. Anything else is a hard error.
      if (resp.status === 507 && !force) {
        const msg = data.detail || 'This model may not fit in the second slot.';
        if (window.confirm(`${msg}\n\nLoad it into Slot B anyway?`)) {
          return loadEngineModelIntoSlotB(modelName, { force: true });
        }
        return;
      }
      throw new Error(data.detail || 'Slot B load failed');
    }
    await refreshModelList();
    await refreshEngineDashboard();
    showToast(`Loaded ${modelName} into Slot B`, 'success');
  } catch (err) {
    showToast(err.message || 'Slot B load failed', 'error');
  }
}

// Unload whatever is in Slot B, freeing its VRAM/RAM.
async function unloadSlotB() {
  try {
    const resp = await fetch('/api/engine/v2/secondary/unload', { method: 'POST' });
    if (!resp.ok) throw new Error('Slot B unload failed');
    await refreshModelList();
    await refreshEngineDashboard();
    showToast('Slot B unloaded', 'success');
  } catch (err) {
    showToast(err.message || 'Slot B unload failed', 'error');
  }
}

// Load (or swap) a model into the managed classifier slot ("Slot C") — the
// resident small workhorse for the classifier/utility (and, when VL+mmproj,
// vision) roles. Swappable with no container recreate; the role resolver
// re-points to it. Mirrors Slot B's admission-retry flow.
async function loadEngineModelIntoSlotC(modelName, { force = false, takeOver = false } = {}) {
  await waitForUiSettingsReady();
  const savedProfile = getEngineModelLoadProfile(modelName) || {};
  try {
    const resp = await fetch('/api/engine/v2/classifier/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelName, ...savedProfile, force, take_over: takeOver }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 507 && !force) {
        const msg = data.detail || 'This model may not fit in the classifier slot.';
        if (window.confirm(`${msg}\n\nLoad it into the classifier slot anyway?`)) {
          return loadEngineModelIntoSlotC(modelName, { force: true, takeOver });
        }
        return;
      }
      if (resp.status === 409 && data.take_over_required && !takeOver) {
        const msg = data.detail || 'The classifier is currently served by the external Docker sidecar.';
        if (window.confirm(`${msg}\n\nTake over the classifier in-app with ${modelName}? (Unloading the slot hands it back to the sidecar.)`)) {
          return loadEngineModelIntoSlotC(modelName, { force, takeOver: true });
        }
        return;
      }
      throw new Error(data.detail || 'Classifier slot load failed');
    }
    const data = await resp.json().catch(() => ({}));
    await refreshModelList();
    await refreshEngineDashboard();
    const visNote = data.vision_capable ? ' (vision)' : '';
    showToast(`Loaded ${modelName} into the classifier slot${visNote}`, 'success');
  } catch (err) {
    showToast(err.message || 'Classifier slot load failed', 'error');
  }
}

// Unload the classifier slot, freeing its VRAM/RAM (roles fall back to primary).
async function unloadSlotC() {
  try {
    const resp = await fetch('/api/engine/v2/classifier/unload', { method: 'POST' });
    if (!resp.ok) throw new Error('Classifier slot unload failed');
    const data = await resp.json().catch(() => ({}));
    await refreshModelList();
    await refreshEngineDashboard();
    showToast(data.restored_external
      ? 'Classifier slot unloaded — external sidecar restored'
      : 'Classifier slot unloaded', 'success');
  } catch (err) {
    showToast(err.message || 'Classifier slot unload failed', 'error');
  }
}

function renderDeviceGrid() {
  const grid = q('mm-device-grid');
  if (!grid) return;

  const caps = getCapabilities();
  const selectedBackend = q('mm-backend-select')?.value;
  const inventory = modalState.inventory || {
    counts: {},
    loadedCounts: {},
    runningOllama: 0,
    engineLoadedModel: '',
  };
  const engineStatus = modalState.engineStatus;
  const lcppStatus = modalState.lcppStatus;
  const extraDetected = (caps.discovered_services || []).filter((service) =>
    !['engine', 'ollama', 'llamacpp'].includes(service.key)
  );
  const engineHasLoadedModel = ['ready', 'starting', 'draining'].includes(engineStatus?.state || '')
    && Boolean(engineStatus?.model_id);

  const cards = [];
  const pushCoreCard = (backend, state) => {
    if (!state.available) return;
    const profile = getBackendProfile(backend);
    const metrics = state.metrics.filter(Boolean)
      .map((metric) => `<span class="mm-device-metric">${escapeHtml(metric)}</span>`)
      .join('');
    const extraActions = (state.actions || []).map((action) => (
      `<button class="btn btn-sm mm-device-secondary${action.className ? ` ${escapeHtml(action.className)}` : ''}" data-device-action="${escapeHtml(action.action)}" data-device-key="${escapeHtml(action.key || backend)}">${escapeHtml(action.label)}</button>`
    )).join('');

    cards.push(`
      <div class="mm-device-card${selectedBackend === backend ? ' selected' : ''}">
        <div class="mm-device-head">
          <div>
            <div class="mm-device-name">${escapeHtml(profile.label)}</div>
            <div class="mm-device-copy">${escapeHtml(profile.description)}</div>
          </div>
          <span class="mm-device-state ${escapeHtml(state.statusClass)}">${escapeHtml(state.statusLabel)}</span>
        </div>
        <div class="mm-device-metrics">${metrics}</div>
        <div class="mm-device-actions">
          <button class="btn btn-sm btn-primary" data-select-backend="${escapeHtml(backend)}">${escapeHtml(profile.actionLabel)}</button>
          ${extraActions}
        </div>
      </div>
    `);
  };

  pushCoreCard('engine', {
    available: caps.has_engine,
    statusClass: engineStatus?.state === 'ready' ? 'good' : engineStatus?.state === 'starting' || engineStatus?.state === 'draining' ? 'busy' : 'muted',
    statusLabel: engineStatus?.state === 'ready' ? 'Ready' : engineStatus?.state === 'starting' ? 'Starting' : engineStatus?.state === 'draining' ? 'Switching' : 'Idle',
    metrics: [
      formatCount(inventory.counts.engine || 0, 'model'),
      engineHasLoadedModel ? `Now using ${formatEngineModelRef(engineStatus.model_id)}` : 'No model loaded right now',
      engineHasLoadedModel && engineStatus?.load_config?.ctx_size ? `Loaded at ${formatTokenCount(engineStatus.load_config.ctx_size)} tokens` : '',
      engineStatus?.gpu?.vram_used_mib ? `VRAM: ${formatBytes(engineStatus.gpu.vram_used_mib * 1024 * 1024)}` : '',
      engineStatus?.ram?.rss_mb ? `System RAM: ${formatBytes(engineStatus.ram.rss_mb * 1024 * 1024)}` : '',
    ],
    actions: [
      { action: 'refresh-setup', label: 'Refresh setup' },
      { action: 'open-advanced', label: 'Advanced' },
    ],
  });

  pushCoreCard('ollama', {
    available: caps.has_ollama,
    statusClass: caps.is_docker_model_runner ? 'muted' : 'good',
    statusLabel: caps.is_docker_model_runner ? 'Docker managed' : 'Available',
    metrics: [
      formatCount(inventory.counts.ollama || 0, 'model'),
      inventory.runningOllama > 0 ? `${formatCount(inventory.runningOllama, 'model')} in memory` : 'Nothing loaded right now',
    ],
  });

  const lcppHealth = lcppStatus?.health?.status || 'unreachable';
  pushCoreCard('llamacpp', {
    available: caps.has_llamacpp,
    statusClass: lcppHealth === 'ok' ? 'good' : lcppHealth === 'loading' || lcppHealth === 'loading model' ? 'busy' : 'muted',
    statusLabel: lcppHealth === 'ok' ? 'Online' : lcppHealth === 'loading' || lcppHealth === 'loading model' ? 'Loading' : 'Offline',
    metrics: [
      formatCount(inventory.counts.llamacpp || 0, 'model'),
      lcppStatus?.props?.default_generation_settings?.model
        ? `Now using ${String(lcppStatus.props.default_generation_settings.model).split('/').pop().replace('.gguf', '')}`
        : 'No model loaded right now',
    ],
    actions: [
      { action: 'open-advanced', label: 'Advanced' },
    ],
  });

  // Always show the vLLM/safetensors device so the format is browsable even
  // before the engine is installed — downloads land on disk and the library
  // notes they need the engine to serve (never a hidden dead-end).
  pushCoreCard('vllm', {
    available: true,
    statusClass: caps.has_vllm ? 'good' : 'muted',
    statusLabel: caps.has_vllm ? 'Available' : 'Not installed',
    metrics: [
      formatCount(inventory.counts.vllm || 0, 'safetensors model'),
      caps.has_vllm ? 'Served by vLLM' : 'Install the vLLM Engine (Discover) to serve',
    ],
    actions: [],
  });

  for (const service of extraDetected) {
    const modelSummary = service.model_count > 0
      ? formatCount(service.model_count, 'model')
      : 'No models detected';
    cards.push(`
      <div class="mm-device-card mm-device-card-detected">
        <div class="mm-device-head">
          <div>
            <div class="mm-device-name">${escapeHtml(service.name)}</div>
            <div class="mm-device-copy">Detected at ${escapeHtml(service.url)}. Manage installs in that app, then Augmentum can use its models.</div>
          </div>
          <span class="mm-device-state muted">Detected</span>
        </div>
        <div class="mm-device-metrics">
          <span class="mm-device-metric">${escapeHtml(modelSummary)}</span>
          <span class="mm-device-metric">${escapeHtml(service.type || 'local app')}</span>
        </div>
        <div class="mm-device-actions">
          <button class="btn btn-sm" data-device-action="hide-detected" data-device-key="${escapeHtml(service.key)}">Hide</button>
        </div>
      </div>
    `);
  }

  grid.innerHTML = cards.length > 0
    ? cards.join('')
    : '<div class="mm-empty">No local model devices were detected yet.</div>';

  grid.querySelectorAll('[data-select-backend]').forEach((btn) => {
    btn.addEventListener('click', () => {
      showDiscoverPane(btn.dataset.selectBackend || 'engine');
    });
  });

  grid.querySelectorAll('[data-device-action="open-advanced"]').forEach((btn) => {
    btn.addEventListener('click', () => {
      setActiveManagerPane('advanced', { expandAdvanced: true });
    });
  });

  grid.querySelectorAll('[data-device-action="refresh-setup"]').forEach((btn) => {
    btn.addEventListener('click', () => {
      refreshDeviceSetup();
    });
  });

  grid.querySelectorAll('[data-device-action="hide-detected"]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      try {
        await fetch('/api/backends/remove', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key: btn.dataset.deviceKey }),
        });
        await fetchCapabilities();
        updateModelManagerVisibility();
        renderDeviceGrid();
      } catch {
        showToast('Could not hide device', 'error');
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Engine v2 Dashboard
// ---------------------------------------------------------------------------

async function refreshEngineDashboard() {
  try {
    const resp = await fetch('/api/engine/v2/status');
    if (!resp.ok) return;
    const status = await resp.json();
    modalState.engineStatus = status;

    // Secondary slot ("Slot B") — a second resident user-chosen model.
    // Cheap, best-effort: a 404 means the feature is off and we just
    // hide the slot UI. Stored so model cards can show a "2nd slot"
    // action and the info line can report its footprint.
    try {
      const secResp = await fetch('/api/engine/v2/secondary/status');
      if (secResp.ok) {
        const sec = await secResp.json();
        modalState.secondaryEnabled = Boolean(sec.enabled);
        modalState.secondary = sec.secondary || null;
        modalState.secondaryModel = sec.secondary?.model_id || '';
      } else {
        modalState.secondaryEnabled = false;
        modalState.secondary = null;
        modalState.secondaryModel = '';
      }
    } catch {
      modalState.secondaryEnabled = false;
    }

    // Managed classifier slot ("Slot C") — swappable classifier/utility/vision
    // model. 404 = feature off → hide its UI. Best-effort, same as Slot B.
    try {
      const clfResp = await fetch('/api/engine/v2/classifier/status');
      if (clfResp.ok) {
        const clf = await clfResp.json();
        modalState.classifierSlotEnabled = Boolean(clf.enabled);
        modalState.classifierSlot = clf.classifier || null;
        modalState.classifierSlotModel = clf.classifier?.model_id || '';
      } else {
        modalState.classifierSlotEnabled = false;
        modalState.classifierSlot = null;
        modalState.classifierSlotModel = '';
      }
    } catch {
      modalState.classifierSlotEnabled = false;
    }

    // Status LED
    const dot = q('mm-engine-led-dot');
    const label = q('mm-engine-led-label');
    if (status.state === 'ready') {
      dot.style.background = 'var(--success)';
      label.textContent = status.model_id || 'Ready';
    } else if (status.state === 'starting' || status.state === 'draining') {
      dot.style.background = 'var(--warning)';
      label.textContent = status.state;
    } else {
      dot.style.background = 'var(--text-muted)';
      label.textContent = 'Idle';
    }

    // Info line
    const info = q('mm-engine-info');
    const parts = [];
    const shouldShowLoadedContext = ['ready', 'starting', 'draining'].includes(status.state || '');
    if (shouldShowLoadedContext && status.profile) {
      parts.push(status.profile.architecture);
      parts.push(status.profile.size_gb + ' GB');
      if (status.profile.is_moe) parts.push('MoE');
    }
    if (shouldShowLoadedContext && status.load_config?.ctx_size) parts.push(`${formatTokenCount(status.load_config.ctx_size)} ctx`);
    if (shouldShowLoadedContext && status.load_config?.kv_cache_type) parts.push(`${String(status.load_config.kv_cache_type).toUpperCase()} KV`);
    if (status.uptime_s) parts.push('up ' + Math.round(status.uptime_s) + 's');
    // Slot B footprint \u2014 show the second resident model + its VRAM so the
    // two-model memory accounting is visible at a glance.
    if (modalState.secondaryEnabled && modalState.secondaryModel) {
      const secVram = modalState.secondary?.actual_memory?.vram_total_mib;
      const secSuffix = secVram ? ` (${(secVram / 1024).toFixed(1)} GB)` : '';
      parts.push(`Slot B: ${modalState.secondaryModel}${secSuffix}`);
    }
    // Slot C footprint \u2014 the resident classifier/utility/vision model.
    if (modalState.classifierSlotEnabled && modalState.classifierSlotModel) {
      const clfVram = modalState.classifierSlot?.actual_memory?.vram_total_mib;
      const clfSuffix = clfVram ? ` (${(clfVram / 1024).toFixed(1)} GB)` : '';
      const visBadge = modalState.classifierSlot?.vision_capable ? ' \ud83d\udc41' : '';
      parts.push(`Slot C: ${modalState.classifierSlotModel}${clfSuffix}${visBadge}`);
    }
    info.textContent = parts.join(' \u2022 ');

    // Model dirs — see /api/engine/v2/models/dirs response shape:
    //   model_dirs: [{path, gguf_count, slow, exists, host_source}]
    //   host_mounts: pending only (active mounts surface in model_dirs
    //     with host_source set)
    //   platform: {id, label, perf_hint}
    const dirsResp = await fetch('/api/engine/v2/models/dirs');
    if (dirsResp.ok) {
      const dirsData = await dirsResp.json();
      const dirs = dirsData.model_dirs || [];
      const hostMounts = dirsData.host_mounts || [];
      const platform = dirsData.platform || {};

      // Platform banner — only shown when there's an actionable hint
      // (Windows/Mac Docker Desktop, or WSL2 with cross-bridge gotchas).
      const platBanner = q('mm-dirs-platform-banner');
      if (platBanner) {
        if (platform.perf_hint) {
          platBanner.classList.remove('hidden');
          platBanner.innerHTML =
            `<div class="mm-platform-banner-label">${escapeHtml(platform.label || 'Host environment')}</div>` +
            `<div class="mm-platform-banner-body">${escapeHtml(platform.perf_hint)}</div>`;
        } else {
          platBanner.classList.add('hidden');
          platBanner.innerHTML = '';
        }
      }

      // Restart banner — actionable when there are pending mounts
      const restartBanner = q('mm-dirs-restart-banner');
      if (restartBanner) {
        if (hostMounts.length > 0) {
          restartBanner.classList.remove('hidden');
          const n = hostMounts.length;
          restartBanner.innerHTML =
            `<strong>${n} ${n === 1 ? 'directory' : 'directories'} waiting</strong> ` +
            '— finish editing <code>compose.yaml</code> and run ' +
            '<code>docker compose restart augmentum</code> to activate.';
        } else {
          restartBanner.classList.add('hidden');
          restartBanner.innerHTML = '';
        }
      }

      // Active list
      const dirsEl = q('mm-engine-dirs');
      if (dirs.length === 0) {
        dirsEl.innerHTML =
          '<div class="mm-dir-empty">' +
            '<div>No directories configured.</div>' +
            '<div class="mm-dir-empty-hint">Add the folder where your GGUFs live to make them visible in the Library.</div>' +
          '</div>';
      } else {
        dirsEl.innerHTML = dirs.map(d => {
          const ggufLabel = d.gguf_count === 1 ? '1 GGUF' : `${d.gguf_count} GGUFs`;
          // Slow pill \u2014 only shown when the mount actually crosses
          // a host bridge / network layer per /proc/self/mountinfo.
          // Tooltip names the actual fs type when known (9p/virtiofs/
          // osxfs/etc.) so power users can verify the diagnosis.
          let slowTip;
          if (d.mount_fs) {
            slowTip = `Mount type: ${d.mount_fs} \u2014 model loads ~10\u00d7 slower than native ext4. Use Localize on slow models, or move the folder to native storage.`;
          } else {
            slowTip = 'Bind mount through a host bridge \u2014 model loads ~10\u00d7 slower than native. Use Localize on slow models, or move the folder to native storage.';
          }
          const slowPill = d.slow
            ? `<span class="mm-status-pill mm-status-slow" title="${escapeHtml(slowTip)}">slow</span>`
            : '';
          const missingPill = !d.exists
            ? '<span class="mm-status-pill mm-status-restart" title="Path is registered but not currently accessible inside the container">missing</span>'
            : '';
          const hostSrc = d.host_source
            ? `<span class="mm-dir-host-source" title="Mounted from host">\u2190 ${escapeHtml(d.host_source)}</span>`
            : '';
          return (
            '<div class="mm-dir-row">' +
              `<code class="mm-dir-path">${escapeHtml(d.path)}</code>` +
              hostSrc +
              `<span class="mm-dir-meta">${ggufLabel}</span>` +
              slowPill +
              missingPill +
              `<button class="mm-dir-remove mm-engine-rm-dir" data-dir="${escapeHtml(d.path)}" title="Remove from scan list">\u00d7</button>` +
            '</div>'
          );
        }).join('');
      }

      dirsEl.querySelectorAll('.mm-engine-rm-dir').forEach(btn => {
        btn.addEventListener('click', async () => {
          try {
            const r = await fetch('/api/engine/v2/models/dirs', {
              method: 'DELETE', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({path: btn.dataset.dir}),
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
          } catch (err) {
            showToast('Remove directory failed: ' + (err?.message || err), 'error');
            console.warn('[models] remove dir failed', err);
            return;
          }
          refreshEngineDashboard();
          refreshModelList();
          fetchModels();
        });
      });

      // Pending list
      const hostEl = q('mm-engine-host-mounts-list');
      const pendingGroup = q('mm-dirs-pending-group');
      if (hostMounts.length > 0) {
        if (pendingGroup) pendingGroup.classList.remove('hidden');
        hostEl.innerHTML = hostMounts.map(m =>
          '<div class="mm-dir-row mm-dir-row-pending">' +
            `<code class="mm-dir-path">${escapeHtml(m.host_path)}</code>` +
            `<span class="mm-dir-meta mm-dir-meta-target">\u2192 ${escapeHtml(m.container_path)}</span>` +
            '<span class="mm-status-pill mm-status-restart" title="Edit compose.yaml then restart augmentum to activate">pending</span>' +
            `<button class="mm-dir-remove mm-host-unmount" data-host="${escapeHtml(m.host_path)}" title="Cancel registration">\u00d7</button>` +
          '</div>'
        ).join('');

        hostEl.querySelectorAll('.mm-host-unmount').forEach(btn => {
          btn.addEventListener('click', async () => {
            try {
              const r = await fetch('/api/engine/v2/models/host-mount', {
                method: 'DELETE', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({host_path: btn.dataset.host}),
              });
              if (!r.ok) throw new Error(`HTTP ${r.status}`);
            } catch (err) {
              showToast('Unmount failed: ' + (err?.message || err), 'error');
              console.warn('[models] host-mount delete failed', err);
              return;
            }
            refreshEngineDashboard();
          });
        });
      } else {
        if (pendingGroup) pendingGroup.classList.add('hidden');
        if (hostEl) hostEl.innerHTML = '';
      }
    }
    renderDeviceGrid();
    renderOverview();
  } catch {
    modalState.engineStatus = null;
    const dot = q('mm-engine-led-dot');
    const label = q('mm-engine-led-label');
    const info = q('mm-engine-info');
    if (dot) dot.style.background = 'var(--text-muted)';
    if (label) label.textContent = 'Idle';
    if (info) info.textContent = '';
    renderDeviceGrid();
    renderOverview();
  }
  refreshKVSettings();
}

const KV_SETTING_INPUTS = [
  { id: 'mm-kv-ttl-days', key: 'engine_kv_ttl_days', type: 'int' },
  { id: 'mm-kv-narrative-ttl-days', key: 'engine_kv_narrative_ttl_days', type: 'int' },
  { id: 'mm-kv-max-snapshots', key: 'engine_kv_max_snapshots_per_model', type: 'int' },
  { id: 'mm-kv-auto-pin-narrative', key: 'engine_kv_auto_pin_narrative', type: 'bool' },
  // Multi-slot KV routing — see
  // docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md. The
  // ``optbool`` type is tri-state (auto/true/false) — see
  // _TRI_STATE_BOOL_SETTINGS in config_routes.py. Persisted None
  // means "follow codebase recommendation"; explicit True/False is
  // a survives-recommendation-change user override. Restart of
  // augmentum required for engine_multislot_enabled changes to flow
  // into the llama-server CLI args.
  { id: 'mm-multislot-enabled', key: 'engine_multislot_enabled', type: 'optbool' },
  { id: 'mm-multislot-parallel', key: 'engine_parallel_slots', type: 'int' },
  { id: 'mm-multislot-cache-ram', key: 'engine_cache_ram_mib', type: 'int' },
  // Engine Defaults section — engine- and hardware-level knobs.
  // The model-level analogues (kv_cache_type, use_jinja_template,
  // reasoning_format) intentionally don't appear here; they live in
  // the per-model Load Options sheet because their right value
  // depends on the loaded model. Backend defaults still exist for
  // these as fallbacks (a model with no Load Options falls back to
  // the engine_* default in config.py); they're just not surfaced
  // in the UI to avoid the "which wins, global or per-model?"
  // confusion. ``int`` covers float on the JS side
  // (``Number(el.value)`` round-trips); the backend casts.
  { id: 'mm-engine-idle-timeout', key: 'engine_idle_timeout', type: 'int' },
  { id: 'mm-engine-warm-on-start', key: 'engine_kv_warm_on_start', type: 'bool' },
  { id: 'mm-engine-flash-attn', key: 'engine_flash_attn', type: 'bool' },
  // MTP self-speculation (upstream PR #22673). Toggle is reload-on-
  // next-model-load (mtp_enabled is snapshotted into CLI args at start).
  // The capability gate in llama_server_manager skips MTP if the loaded
  // GGUF has no built-in heads.
  { id: 'mm-engine-mtp-enabled', key: 'engine_mtp_enabled', type: 'bool' },
  { id: 'mm-engine-mtp-n-max', key: 'engine_mtp_n_max', type: 'int' },
  { id: 'mm-engine-health-timeout', key: 'engine_health_timeout', type: 'int' },
  // Reasoning budget cap — applies live on next request (kwarg is read
  // off settings at request build time in llama_cpp.py + openai_compat.py).
  { id: 'mm-engine-reasoning-budget', key: 'engine_reasoning_budget', type: 'int' },
  { id: 'mm-engine-reasoning-grace', key: 'engine_reasoning_grace_period', type: 'int' },
];

let _kvSettingsBound = false;

async function refreshKVSettings() {
  // Pull current values into the panel. Uses the same /api/config/tools
  // endpoint the rest of the settings UI uses, so admin-gating + DB
  // persistence are already handled by the route layer.
  try {
    const resp = await fetch('/api/config/tools', { credentials: 'same-origin' });
    if (!resp.ok) return;
    const data = await resp.json();
    for (const { id, key, type } of KV_SETTING_INPUTS) {
      const el = document.getElementById(id);
      if (!el || !(key in data)) continue;
      if (type === 'bool') {
        el.checked = !!data[key];
      } else if (type === 'optbool') {
        // Tri-state: data[key] is null | true | false. Map to the
        // <select> option string. Companion ``<key>_resolved`` tells
        // us what auto resolves to right now, used in the label hint.
        const v = data[key];
        el.value = v === null || v === undefined ? 'auto'
                 : v === true ? 'true'
                 : 'false';
        const resolvedKey = `${key}_resolved`;
        const hintEl = document.getElementById(`${id}-resolved-hint`)
                      || document.getElementById(`mm-multislot-resolved-hint`);
        if (hintEl && resolvedKey in data) {
          const resolved = data[resolvedKey] ? 'enabled' : 'disabled';
          hintEl.textContent = v === null
            ? `Auto · currently ${resolved}`
            : (v ? 'Always on' : 'Always off');
        }
      } else if (type === 'string') {
        // Free-form string (or constrained-by-frontend select). The
        // value comes back as a string from the backend's
        // _STRING_SETTINGS table; assign verbatim.
        el.value = data[key] ?? '';
      } else {
        el.value = data[key];
      }
    }
  } catch { /* tolerate transient fetch failure */ }

  if (_kvSettingsBound) return;
  _kvSettingsBound = true;
  for (const { id, key, type } of KV_SETTING_INPUTS) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.addEventListener('change', () => saveKVSetting(key, type, el));
  }
}

async function saveKVSetting(key, type, el) {
  const status = q('mm-kv-status');
  // Coerce per type. ``optbool`` (tri-state): the wire value is
  // null/true/false (not the literal string "auto"); the backend
  // PUT handler also accepts the strings but null is canonical.
  let value;
  if (type === 'bool') {
    value = el.checked;
  } else if (type === 'optbool') {
    const raw = el.value;
    value = raw === 'auto' ? null
          : raw === 'true' ? true
          : raw === 'false' ? false
          : null;
  } else if (type === 'string') {
    value = el.value;
  } else {
    value = Number(el.value);
  }
  try {
    const resp = await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    if (status) {
      status.textContent = 'Saved';
      status.classList.remove('mm-kv-status-err');
      status.classList.add('mm-kv-status-ok');
      setTimeout(() => { if (status.textContent === 'Saved') status.textContent = ''; }, 1500);
    }
    // Re-fetch so the resolved-companion label updates after a tri-state
    // save (the auto/explicit display depends on the persisted value).
    if (type === 'optbool') refreshKVSettings();
  } catch {
    if (status) {
      status.textContent = 'Save failed (admin required?)';
      status.classList.remove('mm-kv-status-ok');
      status.classList.add('mm-kv-status-err');
    }
  }
}

// Wire engine add-dir button + file browser (called once in bindEvents)
function bindEngineEvents() {
  const addBtn = q('mm-engine-add-dir-btn');
  const input = q('mm-engine-add-dir');
  const browseBtn = q('mm-engine-browse-btn');
  if (!addBtn || !input) return;

  // A path that obviously can't exist inside the Linux container —
  // skip the failed-POST-then-fallback round-trip and go straight to
  // host-mount registration. Cuts ~400ms off the add-flow for the
  // most common Windows/WSL case.
  function _looksLikeHostPath(p) {
    return /^[A-Za-z]:[\\/]/.test(p)            // C:\ or C:/
        || p.startsWith('\\\\')                  // UNC: \\wsl$\... or \\server\share
        || p.startsWith('//');                   // alt-form UNC
  }

  async function addDir(path) {
    if (!path) return;
    const instrEl = q('mm-engine-mount-instructions');
    if (instrEl) instrEl.classList.add('hidden');

    // Try as a container path first, unless it's clearly Windows/UNC.
    if (!_looksLikeHostPath(path)) {
      try {
        const r = await fetch('/api/engine/v2/models/dirs', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({path}),
        });
        if (r.ok) {
          input.value = '';
          const browser = q('mm-file-browser');
          if (browser) browser.classList.add('hidden');
          refreshEngineDashboard();
          refreshModelList();
          fetchModels();
          showToast('Folder added \u2014 scanning for models\u2026', 'success');
          return;
        }
      } catch { /* fall through */ }
    }

    // Path doesn't exist in container — treat as host path, register mount
    try {
      const r = await fetch('/api/engine/v2/models/host-mount', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({host_path: path}),
      });
      const data = await r.json();
      if (!r.ok) {
        showToast(data.detail || 'Failed to add directory', 'error');
        return;
      }

      input.value = '';
      refreshEngineDashboard();

      if (data.restart_required && data.volume_line) {
        if (instrEl) {
          instrEl.classList.remove('hidden');
          instrEl.innerHTML =
            '<div class="mm-mount-instr-title">Host folder registered \u2014 one more step</div>' +
            '<div class="mm-mount-instr-body">Open <code>compose.yaml</code>, add this line under <code>augmentum</code> &rsaquo; <code>volumes</code>:</div>' +
            `<code class="mm-mount-instr-snippet">${escapeHtml(data.volume_line)}</code>` +
            '<div class="mm-mount-instr-body">Then restart just Augmentum (other services keep running):</div>' +
            '<code class="mm-mount-instr-snippet">docker compose restart augmentum</code>';
        }
      } else {
        showToast(data.message || 'Already registered', 'info');
      }
    } catch { showToast('Failed to add directory', 'error'); }
  }

  addBtn.addEventListener('click', () => addDir(input.value.trim()));

  // File browser
  if (browseBtn) {
    browseBtn.addEventListener('click', () => {
      const browser = q('mm-file-browser');
      if (browser.classList.contains('hidden')) {
        browser.classList.remove('hidden');
        browseTo('/');
      } else {
        browser.classList.add('hidden');
      }
    });
  }

  async function browseTo(path) {
    const contents = q('mm-fb-contents');
    const breadcrumbs = q('mm-fb-breadcrumbs');
    contents.innerHTML = '<div style="font-size:var(--text-xs);color:var(--text-muted);padding:var(--space-sm)">Loading...</div>';

    try {
      const r = await fetch(`/api/engine/v2/browse?path=${encodeURIComponent(path)}`);
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        contents.innerHTML = `<div style="font-size:var(--text-xs);color:var(--error);padding:var(--space-sm)">${escapeHtml(d.detail || 'Cannot access')}</div>`;
        return;
      }
      const data = await r.json();

      // Breadcrumbs
      breadcrumbs.innerHTML = (data.breadcrumbs || []).map((b, i, arr) =>
        `<button class="mm-fb-crumb" data-path="${escapeHtml(b.path)}" style="background:none;border:none;color:var(--text-link);cursor:pointer;font-size:var(--text-xs);padding:1px 2px">${escapeHtml(b.name)}</button>${i < arr.length - 1 ? '<span style="color:var(--text-muted)">/</span>' : ''}`
      ).join('');
      breadcrumbs.querySelectorAll('.mm-fb-crumb').forEach(btn => {
        btn.addEventListener('click', () => browseTo(btn.dataset.path));
      });

      // Contents
      let html = '';

      // Parent dir
      if (data.parent) {
        html += `<div class="mm-fb-item mm-fb-dir" data-path="${escapeHtml(data.parent)}" style="display:flex;align-items:center;gap:var(--space-xs);padding:4px 6px;cursor:pointer;border-radius:var(--radius-sm)">
          <span style="font-size:var(--text-xs);color:var(--text-muted)">\u2191 ..</span>
        </div>`;
      }

      // Directories
      for (const d of (data.dirs || [])) {
        const badge = d.gguf_count > 0
          ? `<span style="font-size:10px;color:var(--success);margin-left:auto">${d.gguf_count} GGUF${d.gguf_count > 1 ? 's' : ''}</span>`
          : '';
        html += `<div class="mm-fb-item mm-fb-dir" data-path="${escapeHtml(d.path)}" style="display:flex;align-items:center;gap:var(--space-xs);padding:4px 6px;cursor:pointer;border-radius:var(--radius-sm)">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
          <span style="font-size:var(--text-xs)">${escapeHtml(d.name)}</span>
          ${badge}
        </div>`;
      }

      // GGUF files
      for (const f of (data.files || [])) {
        const sizeStr = f.size ? formatBytes(f.size) : '';
        html += `<div class="mm-fb-item" style="display:flex;align-items:center;gap:var(--space-xs);padding:4px 6px;border-radius:var(--radius-sm)">
          <svg viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2" width="14" height="14"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span style="font-size:var(--text-xs)">${escapeHtml(f.name)}</span>
          <span style="font-size:10px;color:var(--text-muted);margin-left:auto">${sizeStr}</span>
        </div>`;
      }

      // "Select this directory" button if GGUFs found
      if ((data.files || []).length > 0) {
        html += `<div style="padding:var(--space-xs);margin-top:var(--space-xs);border-top:1px solid var(--border-light)">
          <button class="btn btn-sm btn-primary mm-fb-select" data-path="${escapeHtml(data.path)}" style="width:100%;font-size:var(--text-xs)">Add this directory (${data.files.length} GGUF${data.files.length > 1 ? 's' : ''})</button>
        </div>`;
      }

      if (!html) {
        html = '<div style="font-size:var(--text-xs);color:var(--text-muted);padding:var(--space-sm)">Empty directory</div>';
      }

      contents.innerHTML = html;

      // Wire clicks
      contents.querySelectorAll('.mm-fb-dir').forEach(el => {
        el.addEventListener('click', () => browseTo(el.dataset.path));
        el.addEventListener('mouseenter', () => el.style.background = 'var(--surface-hover)');
        el.addEventListener('mouseleave', () => el.style.background = '');
      });
      contents.querySelector('.mm-fb-select')?.addEventListener('click', (e) => {
        addDir(e.target.dataset.path);
      });

    } catch {
      contents.innerHTML = '<div style="font-size:var(--text-xs);color:var(--error);padding:var(--space-sm)">Failed to browse</div>';
    }
  }

}

// ---------------------------------------------------------------------------
// HuggingFace Model Search
// ---------------------------------------------------------------------------

async function searchHuggingFace(query, backend) {
  if (hfSearchController) hfSearchController.abort();
  hfSearchController = new AbortController();

  const resultsEl = q('mm-search-results');
  const chipsEl = q('mm-chips');
  const browseEl = q('mm-browse-link');

  // Show loading state, hide chips
  resultsEl.classList.remove('hidden');
  chipsEl.style.display = 'none';
  browseEl.style.display = 'none';
  resultsEl.innerHTML = '<div style="padding:var(--space-sm);font-size:var(--text-xs);color:var(--text-muted)">Searching HuggingFace...</div>';

  const format = backend === 'vllm' ? 'safetensors' : 'gguf';
  try {
    const resp = await fetch(`/api/engine/v2/models/search?q=${encodeURIComponent(query)}&limit=15&format=${format}`, {
      signal: hfSearchController.signal,
    });
    if (!resp.ok) {
      const d = await resp.json().catch(() => ({}));
      resultsEl.innerHTML = `<div style="padding:var(--space-sm);font-size:var(--text-xs);color:var(--error)">${escapeHtml(d.error || 'Search failed')}</div>`;
      return;
    }
    const data = await resp.json();
    const results = data.results || [];

    if (results.length === 0) {
      const noun = format === 'safetensors' ? 'safetensors' : 'GGUF';
      resultsEl.innerHTML = `<div style="padding:var(--space-sm);font-size:var(--text-xs);color:var(--text-muted)">No ${noun} models found.</div>`;
      return;
    }

    // Cache the fetched results so "Back to results" in the file picker can
    // re-show them without a re-fetch or the user re-typing the query.
    _lastHfSearch = { results, backend };
    renderHfSearchResults(results, backend);

  } catch (err) {
    if (err.name === 'AbortError') return;
    resultsEl.innerHTML = `<div style="padding:var(--space-sm);font-size:var(--text-xs);color:var(--error)">Search failed: ${escapeHtml(err.message)}</div>`;
  }
}

// Render (and wire) a set of HF search results into the results pane. Split
// out of searchHuggingFace so the file-picker "Back to results" button can
// re-render the last set from cache \u2014 no network, no re-typing.
function renderHfSearchResults(results, backend) {
  const resultsEl = q('mm-search-results');
  if (!resultsEl) return;
  const isVllm = backend === 'vllm';

  // vLLM: safetensors repos download whole, so no per-quant chips — each result
  // gets one "Download repo" action, with a shared destination picker (multi-
  // drive) at the top so the user chooses which drive (never forced to C:).
  if (isVllm) {
    const destRow = `<div class="mm-st-dest-row" style="display:flex;align-items:center;gap:8px;padding:6px var(--space-sm);font-size:var(--text-xs);color:var(--text-muted)">
      <span>Download to</span>
      <select id="mm-st-dest" class="field-input" style="flex:1;min-width:0"></select>
    </div>`;
    const items = results.map(r => {
      const downloads = r.downloads >= 1000
        ? (r.downloads / 1000).toFixed(r.downloads >= 10000 ? 0 : 1) + 'k'
        : String(r.downloads || 0);
      const files = r.files || [];
      const totalSize = files.reduce((s, f) => s + (f.size || 0), 0);
      const sizeLabel = totalSize > 0 ? formatModelFileSize(totalSize) : `${files.length} files`;
      const tags = [];
      if (r.tags && r.tags.some(t => /vl|vision|image/i.test(t))) tags.push('Vision');
      const installed = isModelInstalled(r.id);
      const installedBadge = installed ? '<span class="mm-search-installed">✓ installed</span>' : '';
      return `<div class="mm-search-item mm-st-item" data-repo="${escapeHtml(r.id)}" data-backend="vllm"
                   data-files="${escapeHtml(JSON.stringify(files.map(f => f.name)))}" data-installed="${installed ? '1' : '0'}"
                   style="cursor:${installed ? 'default' : 'pointer'};user-select:none" title="${installed ? 'Already in your library' : 'Click to download this repo'}">
        <div class="mm-search-item-row">
          <span class="mm-search-repo">${escapeHtml(r.id)}</span>
          ${installedBadge}
          <span class="mm-search-counter" title="${r.downloads} downloads">⬇ ${downloads}</span>
        </div>
        <div class="mm-search-quants">
          <button class="mm-search-quant mm-st-download" data-repo="${escapeHtml(r.id)}"
                  data-files="${escapeHtml(JSON.stringify(files.map(f => f.name)))}"
                  ${installed ? 'disabled' : ''}>
            ${installed ? 'In library' : 'Download repo'} <span class="mm-search-quant-size">${escapeHtml(sizeLabel)}</span>
          </button>
          ${tags.map(t => `<span class="mm-search-no-files">${escapeHtml(t)}</span>`).join('')}
        </div>
      </div>`;
    }).join('');
    resultsEl.innerHTML = destRow + items;
    // Populate the shared destination picker (multi-drive).
    populateSafetensorsDestinations();
    // Whole row is clickable (like the GGUF results) — not just the small button.
    const triggerDownload = (el) => {
      if (!el || el.dataset.installed === '1') return;
      let files = [];
      try { files = JSON.parse(el.dataset.files || '[]'); } catch { /* ignore */ }
      const dest = q('mm-st-dest')?.value || '';
      enqueueSafetensors(el.dataset.repo, files, dest);
    };
    resultsEl.querySelectorAll('.mm-st-item').forEach((item) => {
      item.addEventListener('click', () => triggerDownload(item));
    });
    resultsEl.querySelectorAll('.mm-st-download').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        triggerDownload(btn.closest('.mm-st-item'));
      });
    });
    return;
  }

  resultsEl.innerHTML = results.map(r => {
    const downloads = r.downloads >= 1000
      ? (r.downloads / 1000).toFixed(r.downloads >= 10000 ? 0 : 1) + 'k'
      : String(r.downloads || 0);
    const fileCount = (r.files || []).length;
    const fileHint = fileCount > 0 ? `${fileCount} file${fileCount > 1 ? 's' : ''}` : 'no files listed';

    // Show top quant files inline (max 4)
    const topFiles = (r.files || [])
      .sort((a, b) => {
        const aUnknown = a.size > 0 ? 0 : 1;
        const bUnknown = b.size > 0 ? 0 : 1;
        if (aUnknown !== bUnknown) return aUnknown - bUnknown;
        if (a.size > 0 && b.size > 0) return a.size - b.size;
        return a.name.localeCompare(b.name);
      })
      .slice(0, 4);
    const quantChips = topFiles.map(f => {
      // Extract quant from filename (e.g. Q4_K_M from Model-Q4_K_M.gguf)
      const m = f.name.match(/[_.-]((?:IQ|Q|F|BF)\d[^\s.]*)/i);
      const label = m ? m[1] : f.name.replace(/\.gguf$/i, '').slice(-12);
      const sizeLabel = formatModelFileSize(f.size);
      return `<button class="mm-search-quant" data-repo="${escapeHtml(r.id)}" data-file="${escapeHtml(f.name)}" data-backend="${escapeHtml(backend)}" title="${escapeHtml(f.name)} (${escapeHtml(sizeLabel)})">${escapeHtml(label)} <span class="mm-search-quant-size">${escapeHtml(sizeLabel)}</span></button>`;
    }).join('');

    const installed = isModelInstalled(r.id);
    const installedBadge = installed
      ? '<span class="mm-search-installed" title="Already in your library">\u2713 installed</span>'
      : '';
    return `<div class="mm-search-item${installed ? ' mm-search-item-installed' : ''}" data-repo="${escapeHtml(r.id)}" data-backend="${escapeHtml(backend)}">
      <div class="mm-search-item-row">
        <span class="mm-search-repo">${escapeHtml(r.id)}</span>
        ${installedBadge}
        <span class="mm-search-counter" title="${r.downloads} downloads">\u2b07 ${downloads}</span>
        <span class="mm-search-counter" title="${r.likes} likes">\u2764 ${r.likes || 0}</span>
      </div>
      <div class="mm-search-quants">
        ${quantChips}
        ${fileCount > 4 ? `<button class="mm-search-more" data-repo="${escapeHtml(r.id)}" data-backend="${escapeHtml(backend)}">+${fileCount - 4} more\u2026</button>` : ''}
        ${fileCount === 0 ? `<span class="mm-search-no-files">${fileHint}</span>` : ''}
      </div>
    </div>`;
  }).join('');

  // Wire click handlers
  // Clicking a quant chip directly starts the GGUF file picker or download
  resultsEl.querySelectorAll('.mm-search-quant').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const dlBackend = btn.dataset.backend;
      const repo = btn.dataset.repo;
      const file = btn.dataset.file;
      hideSearchResults();
      q('mm-pull-input').value = repo;
      updatePullClearVisibility();
      if (dlBackend === 'engine') {
        pullModel(`${repo}:${file}`, 'engine');
      } else {
        pullGgufModel(repo, file);
      }
    });
  });

  // Clicking "more" or the row itself opens the full GGUF file picker.
  // `fromSearch` tells the picker to show a "Back to results" button.
  resultsEl.querySelectorAll('.mm-search-more').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const repo = btn.dataset.repo;
      const dlBackend = btn.dataset.backend;
      hideSearchResults();
      q('mm-pull-input').value = repo;
      updatePullClearVisibility();
      fetchGgufFiles(repo, dlBackend, { fromSearch: true });
    });
  });
  resultsEl.querySelectorAll('.mm-search-item').forEach(item => {
    item.addEventListener('click', () => {
      const repo = item.dataset.repo;
      const dlBackend = item.dataset.backend;
      hideSearchResults();
      q('mm-pull-input').value = repo;
      updatePullClearVisibility();
      fetchGgufFiles(repo, dlBackend, { fromSearch: true });
    });
  });
}

// Re-show the last search results (from the file picker's Back button).
// Hides the picker and re-hides the recommended chips, mirroring the state
// searchHuggingFace leaves behind. Returns false if there's nothing cached.
function showLastHfSearchResults() {
  if (!_lastHfSearch) return false;
  const picker = q('mm-gguf-picker');
  if (picker) picker.classList.add('hidden');
  const chipsEl = q('mm-chips');
  const browseEl = q('mm-browse-link');
  if (chipsEl) chipsEl.style.display = 'none';
  if (browseEl) browseEl.style.display = 'none';
  const resultsEl = q('mm-search-results');
  if (resultsEl) resultsEl.classList.remove('hidden');
  renderHfSearchResults(_lastHfSearch.results, _lastHfSearch.backend);
  return true;
}

// Create/toggle the file picker's "Back to results" button. It's created once
// and reused across picker opens (idempotent) so we never stack duplicates.
function syncGgufBackButton(picker, show) {
  if (!picker) return;
  let backBtn = q('mm-gguf-back');
  if (show) {
    if (!backBtn) {
      backBtn = document.createElement('button');
      backBtn.id = 'mm-gguf-back';
      backBtn.type = 'button';
      backBtn.className = 'btn btn-sm mm-gguf-back';
      backBtn.style.marginBottom = 'var(--space-xs)';
      backBtn.style.alignSelf = 'flex-start';
      backBtn.innerHTML = '← Back to results';
      backBtn.addEventListener('click', () => { showLastHfSearchResults(); });
      picker.insertBefore(backBtn, picker.firstChild);
    }
    backBtn.classList.remove('hidden');
  } else if (backBtn) {
    backBtn.classList.add('hidden');
  }
}

// Show the input's clear (×) affordance only when there's text to clear, so it
// can never be mis-clicked on an empty box.
function updatePullClearVisibility() {
  const input = q('mm-pull-input');
  const clearBtn = q('mm-pull-clear');
  if (!input || !clearBtn) return;
  clearBtn.classList.toggle('hidden', !input.value.trim());
}

function hideSearchResults() {
  const resultsEl = q('mm-search-results');
  if (resultsEl) {
    resultsEl.classList.add('hidden');
    resultsEl.innerHTML = '';
  }
  // Restore chips and browse link
  const chipsEl = q('mm-chips');
  const browseEl = q('mm-browse-link');
  if (chipsEl) chipsEl.style.display = '';
  if (browseEl) browseEl.style.display = '';
}

// ---------------------------------------------------------------------------
// Model List
// ---------------------------------------------------------------------------

// Render the downloaded safetensors repos (vLLM engine) as their own library
// group — capability badges (arch/context/params/vision/MoE/remote-code) +
// registered status + delete. Appended after the GGUF/engine groups so a user
// can see and manage safetensors models exactly like GGUFs.
// Format filter for the library (GGUF vs Safetensors). Only shown when both
// are present. Groups are tagged with data-mm-format; the chips toggle group
// visibility. State persists across refreshes within the session.
let _mmFormatFilter = 'all';

function applyFormatFilter(list) {
  list.querySelectorAll('[data-mm-format]').forEach((g) => {
    const fmt = g.dataset.mmFormat;
    g.style.display = (_mmFormatFilter === 'all' || _mmFormatFilter === fmt) ? '' : 'none';
  });
}

function maybeAddFormatFilter(list) {
  const hasGguf = list.querySelector('[data-mm-format="gguf"]');
  const hasSt = list.querySelector('[data-mm-format="safetensors"]');
  // Only one format present → nothing to filter. Reset to 'all' so a stale
  // filter (e.g. 'safetensors' after the last safetensors model was deleted)
  // can never hide the remaining groups and blank the library.
  if (!hasGguf || !hasSt) { _mmFormatFilter = 'all'; applyFormatFilter(list); return; }
  const bar = document.createElement('div');
  bar.className = 'mm-format-filter';
  bar.style.cssText = 'display:flex;gap:6px;padding:6px var(--space-sm);align-items:center';
  const chip = (val, label) => `<button class="mm-chip${_mmFormatFilter === val ? ' mm-chip-installed' : ''}" data-fmt="${val}">${label}</button>`;
  bar.innerHTML = `<span style="font-size:var(--text-xs);color:var(--text-muted);margin-right:2px">Format</span>`
    + chip('all', 'All') + chip('gguf', 'GGUF') + chip('safetensors', 'Safetensors');
  bar.querySelectorAll('[data-fmt]').forEach((b) => b.addEventListener('click', () => {
    _mmFormatFilter = b.dataset.fmt;
    bar.querySelectorAll('[data-fmt]').forEach((x) => x.classList.toggle('mm-chip-installed', x.dataset.fmt === _mmFormatFilter));
    applyFormatFilter(list);
  }));
  list.insertBefore(bar, list.firstChild);
  applyFormatFilter(list);
}

// Auto-detect safetensors models dropped into a model dir while the manager is
// open — no manual refresh, no restart. Polls the live filesystem scan and only
// re-renders when the set actually changes (cheap: os.scandir on the model dirs).
let _stWatchTimer = null;
let _stWatchSig = null;

function _stSig(repos) {
  return (repos || [])
    .map((r) => `${r.name}:${r.registered ? 1 : 0}:${r.size || 0}`)
    .sort().join('|');
}

async function _pollSafetensorsLibrary() {
  if (!modalEl || modalState.activePane !== 'library') return;
  try {
    const resp = await fetch('/api/models/safetensors/local');
    if (!resp.ok) return;
    const repos = (await resp.json()).repos || [];
    const sig = _stSig(repos);
    if (_stWatchSig !== null && sig !== _stWatchSig) {
      _stWatchSig = sig;
      await refreshModelList(); // a model was added/removed on disk — reflect it
    } else {
      _stWatchSig = sig;
    }
  } catch { /* transient — next tick retries */ }
}

function startSafetensorsWatch() {
  stopSafetensorsWatch();
  _stWatchTimer = setInterval(_pollSafetensorsLibrary, 4000);
}

function stopSafetensorsWatch() {
  if (_stWatchTimer) { clearInterval(_stWatchTimer); _stWatchTimer = null; }
}

async function renderSafetensorsLibrary(list) {
  let repos = [];
  try {
    const resp = await fetch('/api/models/safetensors/local');
    if (resp.ok) repos = (await resp.json()).repos || [];
  } catch { return; }
  // Baseline for the auto-detect watcher = what we're about to render.
  _stWatchSig = _stSig(repos);

  // Feed the vLLM device-card count.
  if (modalState.inventory) {
    modalState.inventory.counts = modalState.inventory.counts || {};
    modalState.inventory.counts.vllm = repos.length;
    renderDeviceGrid();
  }
  if (!repos.length) return;

  const fmtParams = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${Math.round(n / 1e6)}M` : '');
  const badge = (t, warn) => `<span class="mm-device-metric"${warn ? ' style="color:var(--warning,#e6a817)"' : ''}>${escapeHtml(t)}</span>`;
  const cards = repos.map((r) => {
    const badges = [];
    if (r.architecture) badges.push(badge(r.architecture));
    if (r.context_length) badges.push(badge(`${formatTokenCount(r.context_length)} ctx`));
    if (r.params_est) badges.push(badge(fmtParams(r.params_est)));
    if (r.is_moe) badges.push(badge(`MoE ${r.expert_count}`));
    if (r.vision) badges.push(badge('Vision'));
    if (r.tools) badges.push(badge('Tools'));
    if (r.needs_remote_code) badges.push(badge('⚠ remote-code', true));
    badges.push(badge(r.registered ? '● registered' : 'not registered'));
    return `<div class="mm-model-card" style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;padding:8px var(--space-sm);border-bottom:1px solid var(--border,rgba(128,128,128,0.15))">
      <div style="min-width:0">
        <div class="mm-model-name" style="font-weight:600">${escapeHtml(r.name)}</div>
        <div class="mm-device-metrics" style="margin-top:4px;flex-wrap:wrap">${badges.join('')}</div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
        <span style="font-size:var(--text-xs);color:var(--text-muted)">${escapeHtml(formatModelFileSize(r.size || 0))}</span>
        ${r.registered
          // Installed by Augmentum (has a llama-swap registration).
          ? `<button class="btn btn-sm btn-primary mm-st-load" data-model="${escapeHtml(r.model_name || r.name)}">Load</button>
             <button class="btn btn-sm mm-st-edit" data-path="${escapeHtml(r.path)}" data-name="${escapeHtml(r.name)}">Launch settings</button>
             <button class="btn btn-sm mm-st-delete" data-registered="1" data-path="${escapeHtml(r.path)}" data-name="${escapeHtml(r.name)}">Delete</button>`
          // Found on disk but NOT installed through Augmentum (could be a training
          // artifact or a manual download). Keep management — offer to register it
          // for serving — but delete carries a STRONGER confirmation since it
          // wasn't Augmentum's to install.
          : `<span style="font-size:var(--text-xs);color:var(--text-muted);font-style:italic" title="Not installed through Augmentum">detected on disk</span>
             <button class="btn btn-sm mm-st-edit" data-path="${escapeHtml(r.path)}" data-name="${escapeHtml(r.name)}">Register to serve</button>
             <button class="btn btn-sm mm-st-delete" data-registered="0" data-path="${escapeHtml(r.path)}" data-name="${escapeHtml(r.name)}">Delete</button>`}
      </div>
    </div>`;
  }).join('');

  const hasEngine = getCapabilities().has_vllm;
  const group = document.createElement('div');
  group.className = 'mm-model-group';
  group.dataset.mmFormat = 'safetensors';
  const copy = hasEngine
    ? escapeHtml(formatCount(repos.length, 'safetensors model'))
    : `${escapeHtml(formatCount(repos.length, 'safetensors model'))} · install the vLLM Engine to serve these`;
  const addBtn = hasEngine
    ? '<button class="btn btn-sm mm-group-select-btn" data-select-backend="vllm">Add here</button>'
    : '';
  group.innerHTML = `
    <div class="mm-model-group-header">
      <div>
        <div class="mm-model-group-title">${hasEngine ? 'vLLM Engine' : 'Safetensors Models'}</div>
        <div class="mm-model-group-copy">${copy}</div>
      </div>
      ${addBtn}
    </div>
    <div class="mm-model-group-list">${cards}</div>`;
  group.querySelector('.mm-group-select-btn')?.addEventListener('click', () => showDiscoverPane('vllm'));
  group.querySelectorAll('.mm-st-delete').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const { path, name, registered } = btn.dataset;
      // Stronger confirmation for models NOT installed through Augmentum — they
      // may be manual downloads or training artifacts the user placed by hand.
      const msg = registered === '0'
        ? `⚠ "${name}" was NOT installed through Augmentum — it's on disk from a manual add or training run.\n\nPermanently delete all its files? This cannot be undone.`
        : `Delete ${name}? Removes the model files and unregisters it from the vLLM engine.`;
      if (!window.confirm(msg)) return;
      btn.disabled = true;
      try {
        const resp = await fetch('/api/models/safetensors/local', {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path }),
        });
        const data = await resp.json().catch(() => ({}));
        if (resp.ok) { showToast(`Deleted ${name}`, 'success'); await refreshModelList(); }
        else { btn.disabled = false; showToast(`Delete failed: ${data.detail || data.error || resp.status}`, 'error'); }
      } catch (err) { btn.disabled = false; showToast(`Delete failed: ${err.message || err}`, 'error'); }
    });
  });
  group.querySelectorAll('.mm-st-edit').forEach((btn) => {
    btn.addEventListener('click', () => openSafetensorsLaunchEditor(btn.dataset.path, btn.dataset.name));
  });
  group.querySelectorAll('.mm-st-load').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const model = btn.dataset.model;
      const orig = btn.textContent;
      btn.disabled = true; btn.textContent = 'Loading…';
      showToast(`Loading ${model} into vLLM — first load can take a minute…`, 'info');
      try {
        const resp = await fetch('/api/models/vllm/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_name: model }),
        });
        const d = await resp.json().catch(() => ({}));
        if (resp.ok) showToast(`${model} loaded — select it in the chat model picker to use it`, 'success');
        else showToast(`Load failed: ${d.detail || d.error || `HTTP ${resp.status}`}`, 'error');
      } catch (err) {
        showToast(`Load failed: ${err.message || err}`, 'error');
      }
      btn.disabled = false; btn.textContent = orig;
    });
  });
  list.appendChild(group);
}

// Per-model vLLM launch-params editor (dtype, context, GPU util, tensor-parallel,
// trust-remote-code, quantization). Loads derived defaults + saved overrides,
// writes back via PUT which rewrites the llama-swap entry (engine auto-reloads).
async function openSafetensorsLaunchEditor(path, name) {
  let data = null;
  try {
    const resp = await fetch(`/api/models/safetensors/local/launch?path=${encodeURIComponent(path)}`);
    if (!resp.ok) { showToast('Could not load launch settings', 'error'); return; }
    data = await resp.json();
  } catch { showToast('Could not load launch settings', 'error'); return; }

  const d = data.derived || {};
  const p = data.params || {};
  const val = (k) => (p[k] !== undefined && p[k] !== null ? p[k] : d[k]);
  const maxCtx = data.model_max_context || 0;

  const scrim = document.createElement('div');
  scrim.className = 'discover-svc-scrim';
  scrim.style.cssText = 'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.5);backdrop-filter:blur(2px)';
  const field = (label, inner, hint) => `<label style="display:block;margin:10px 0;font-size:var(--text-xs)">
      <span style="display:block;color:var(--text-muted);margin-bottom:4px">${escapeHtml(label)}</span>${inner}
      ${hint ? `<span style="display:block;color:var(--text-muted);opacity:0.7;margin-top:3px">${escapeHtml(hint)}</span>` : ''}
    </label>`;
  const dtypes = ['bfloat16', 'float16', 'float32', 'auto'];
  scrim.innerHTML = `
    <div role="dialog" aria-label="${escapeHtml(name)} launch settings" style="background:var(--surface-1,#16161c);color:var(--text,inherit);border:1px solid rgba(128,128,128,0.28);border-radius:14px;padding:20px 22px;max-width:440px;width:90%;box-shadow:0 14px 48px rgba(0,0,0,0.45);max-height:85vh;overflow:auto">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
        <strong>${escapeHtml(name)} — launch settings</strong>
        <button class="stl-close" style="border:none;background:transparent;color:inherit;opacity:0.6;font-size:1.3em;cursor:pointer">×</button>
      </div>
      ${field('Precision (dtype)', `<select class="field-input stl-dtype">${dtypes.map((t) => `<option value="${t}"${val('dtype') === t ? ' selected' : ''}>${t}</option>`).join('')}</select>`)}
      ${field('Context length (max-model-len)', `<input class="field-input stl-ctx" type="number" min="512" step="512" value="${escapeHtml(String(val('max_model_len') || 8192))}">`, maxCtx ? `Model supports up to ${maxCtx.toLocaleString()} tokens. Larger context needs more VRAM for KV.` : '')}
      ${field('GPU memory fraction', `<input class="field-input stl-gpu" type="number" min="0.1" max="0.98" step="0.05" value="${escapeHtml(String(val('gpu_memory_utilization') || 0.9))}">`, 'Share of GPU VRAM vLLM may reserve.')}
      ${field('Tensor parallel (GPUs)', `<input class="field-input stl-tp" type="number" min="1" max="8" step="1" value="${escapeHtml(String(val('tensor_parallel_size') || 1))}">`, 'Split across N GPUs (multi-GPU boxes).')}
      ${field('Quantization', `<input class="field-input stl-quant" type="text" placeholder="none (e.g. awq, gptq, fp8)" value="${escapeHtml(String(val('quantization') || ''))}">`)}
      <label style="display:flex;align-items:center;gap:8px;margin:12px 0;font-size:var(--text-xs)">
        <input type="checkbox" class="stl-trust"${val('trust_remote_code') ? ' checked' : ''}>
        <span>Trust remote code ${data.profile?.needs_remote_code ? '<b>(this model requires it)</b>' : "(runs the repo's Python — enable only if you trust it)"}</span>
      </label>
      <div class="stl-feedback" style="font-size:var(--text-xs);min-height:1.1em;opacity:0.85;margin:6px 0"></div>
      <div style="display:flex;gap:8px;margin-top:8px">
        <button class="btn btn-sm btn-primary stl-save" style="flex:1">Save &amp; apply</button>
        <button class="btn btn-sm stl-cancel">Cancel</button>
      </div>
    </div>`;
  (window._mmOverlay || document.body).appendChild(scrim);
  const close = () => scrim.remove();
  scrim.addEventListener('click', (e) => { if (e.target === scrim) close(); });
  scrim.querySelector('.stl-close').addEventListener('click', close);
  scrim.querySelector('.stl-cancel').addEventListener('click', close);
  scrim.querySelector('.stl-save').addEventListener('click', async () => {
    const params = {
      dtype: scrim.querySelector('.stl-dtype').value,
      max_model_len: parseInt(scrim.querySelector('.stl-ctx').value, 10) || 8192,
      gpu_memory_utilization: parseFloat(scrim.querySelector('.stl-gpu').value) || 0.9,
      tensor_parallel_size: parseInt(scrim.querySelector('.stl-tp').value, 10) || 1,
      quantization: scrim.querySelector('.stl-quant').value.trim(),
      trust_remote_code: scrim.querySelector('.stl-trust').checked,
    };
    const fb = scrim.querySelector('.stl-feedback');
    fb.textContent = 'Saving…';
    try {
      const resp = await fetch('/api/models/safetensors/local/launch', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, params }),
      });
      const rd = await resp.json().catch(() => ({}));
      if (resp.ok) {
        showToast(`${name}: launch settings applied`, 'success');
        close();
        await refreshModelList();
      } else {
        fb.textContent = `✗ ${rd.detail || rd.error || `Failed (${resp.status})`}`;
      }
    } catch (err) { fb.textContent = `✗ ${err.message || err}`; }
  });
}

async function refreshModelList() {
  const list = q('mm-model-list');
  list.innerHTML = '<div class="mm-empty">Loading models...</div>';

  try {
    const caps = getCapabilities();
    const fetches = [
      getAllModels(),
      fetch('/api/ps').catch(() => null),
    ];
    // If llama.cpp is available, also fetch router models for status
    if (caps.has_llamacpp) {
      fetches.push(fetch('/api/llamacpp/status').catch(() => null));
    } else {
      fetches.push(null);
    }
    // Engine v2 status
    if (caps.has_engine) {
      fetches.push(fetch('/api/engine/v2/status').catch(() => null));
    } else {
      fetches.push(null);
    }
    // Per-drive disk usage for the storage rollup + filter chips.
    fetches.push(fetch('/api/models/storage').catch(() => null));

    const [allModels, runningResp, lcppResp, engineResp, storageResp] = await Promise.all(fetches);

    // Engine catalog carries each model's on-disk path + the configured
    // model_dirs — needed to label which drive a model lives on. Best-effort:
    // if it's mid-scan the drive badges simply don't render this pass.
    if (caps.has_engine) {
      await refreshEngineModelCatalog().catch(() => []);
    }
    if (storageResp && storageResp.ok) {
      const storageData = await storageResp.json().catch(() => ({}));
      modalState.driveStorage = storageData.destinations || [];
    }
    modalState.roleOptions = buildRoleOptions(allModels);
    renderRoleSettings();

    // Engine v2 loaded model
    let engineLoadedModel = '';
    let engineState = 'idle';
    if (engineResp && engineResp.ok) {
      const engineData = await engineResp.json();
      engineLoadedModel = engineData.model_id || '';
      engineState = engineData.state || 'idle';
    }

    const models = allModels.filter((m) => {
      // Hide mode-prefixed virtual models
      if (m.name.startsWith('a/') || m.name.startsWith('n/') || m.name.startsWith('p/')) return false;
      // Show manageable backends (ollama, llamacpp, engine)
      const backend = m.details?.augmentum_backend || 'ollama';
      return backend === 'ollama' || backend === 'llamacpp' || backend === 'engine';
    });
    modalState.managedModels = models;

    let runningModels = [];
    let runningModelsData = [];
    if (runningResp && runningResp.ok) {
      const runningData = await runningResp.json();
      runningModelsData = runningData.models || [];
      runningModels = runningModelsData.map((m) => m.name);
    }

    // Build router model status map (model name → status string)
    let routerStatusMap = {};
    let isRouterMode = false;
    if (lcppResp && lcppResp.ok) {
      const lcppData = await lcppResp.json();
      isRouterMode = lcppData.is_router_mode || false;
      if (isRouterMode) {
        for (const rm of (lcppData.router_models || [])) {
          const name = rm.model || rm.id || '';
          routerStatusMap[name] = rm.status || 'unloaded';
        }
      }
    }

    if (models.length === 0) {
      modalState.inventory = {
        counts: { engine: 0, ollama: 0, llamacpp: 0 },
        loadedCounts: { engine: 0, ollama: 0, llamacpp: 0 },
        runningOllama: 0,
        engineLoadedModel,
      };
      modalState.modelEntries = emptyModelEntries();
      renderDeviceGrid();
      renderRoleSettings();
      renderOverview();
      list.innerHTML = `
        <div class="mm-empty-welcome">
          <p>No models are installed yet. Choose a device above, then add a model to get started.</p>
          <div class="mm-chips">
            <button class="mm-chip" data-model="llama3.1:8b">llama3.1:8b</button>
            <button class="mm-chip" data-model="mistral">mistral</button>
            <button class="mm-chip" data-model="gemma2">gemma2</button>
          </div>
        </div>
      `;
      list.querySelectorAll('.mm-chip').forEach((btn) => {
        btn.addEventListener('click', () => pullModel(btn.dataset.model));
      });
      // Even with no GGUF/engine/ollama models, the user may have safetensors
      // repos on disk — show them so they're never invisible.
      await renderSafetensorsLibrary(list).catch(() => {});
      return;
    }

    const groups = {
      engine: [],
      ollama: [],
      llamacpp: [],
    };
    const modelEntries = emptyModelEntries();
    const entryByName = new Map();
    const inventory = {
      counts: { engine: 0, ollama: 0, llamacpp: 0 },
      loadedCounts: { engine: 0, ollama: 0, llamacpp: 0 },
      runningOllama: runningModels.length,
      engineLoadedModel,
    };

    list.innerHTML = '';
    // Drive lookup is only meaningful when models span >1 mounted folder.
    const driveMap = hasMultipleDrives() ? buildDriveByModelName() : new Map();
    for (const model of models) {
      const backend = (model.details && model.details.augmentum_backend) || 'ollama';
      const isOllama = backend === 'ollama';
      const isLlamacpp = backend === 'llamacpp';
      const isEngine = backend === 'engine';
      const isLoaded = isOllama && runningModels.some(
        (r) => r === model.name || r.startsWith(model.name.split(':')[0])
      );
      // For llama.cpp router mode, get model-specific status
      const routerStatus = (isLlamacpp && isRouterMode) ? (routerStatusMap[model.name] || 'loaded') : null;
      const runInfo = runningModelsData.find(r => r.name === model.name || r.name?.startsWith(model.name.split(':')[0]));
      // Engine: check if this model is the currently loaded one
      const engineLoaded = isEngine && engineLoadedModel === model.name;
      const effectiveBackend = groups[backend] ? backend : 'ollama';
      const entryData = {
        model,
        isLoaded: isOllama ? isLoaded : isEngine ? engineLoaded : routerStatus === 'loaded',
        isSelected: window.app?.state?.currentModel === model.name,
        isOllama,
        routerStatus,
        runningInfo: runInfo || null,
        isEngine,
        engineLoaded,
        engineState,
      };
      inventory.counts[effectiveBackend] = (inventory.counts[effectiveBackend] || 0) + 1;
      if (isOllama && isLoaded) inventory.loadedCounts.ollama += 1;
      if (isEngine && engineLoaded) inventory.loadedCounts.engine += 1;
      if (isLlamacpp && routerStatus === 'loaded') inventory.loadedCounts.llamacpp += 1;

      const driveDir = isEngine ? driveForModelName(model.name, driveMap) : null;
      const card = createModelCard(
        model,
        isLoaded,
        isOllama,
        routerStatus,
        runInfo || null,
        isEngine,
        engineLoaded,
        engineState,
        driveDir,
      );
      const entry = {
        model,
        isLoaded: entryData.isLoaded,
        isSelected: entryData.isSelected,
        card,
      };
      groups[effectiveBackend].push(entry);
      entryByName.set(model.name, entry);
      if (modelEntries[effectiveBackend]) modelEntries[effectiveBackend].push(entryData);
    }

    modalState.inventory = inventory;
    modalState.modelEntries = modelEntries;
    renderDeviceGrid();
    renderOverview();

    // Drive filter bar — only when models span multiple mounted folders.
    // Each chip narrows the list to one drive; "All drives" clears it.
    if (hasMultipleDrives()) {
      const countByDir = new Map();
      for (const entry of groups.engine) {
        const d = entry.card.dataset.drive || '';
        if (d) countByDir.set(d, (countByDir.get(d) || 0) + 1);
      }
      const storageByDir = new Map((modalState.driveStorage || [])
        .map((s) => [String(s.dir || '').replace(/\/+$/, ''), s]));
      const chips = ['<button class="mm-drive-chip" data-drive-chip="__all__">All drives</button>'];
      for (const dir of engineModelDirs()) {
        const d = String(dir).replace(/\/+$/, '');
        const label = driveLabelFromDir(d);
        const n = countByDir.get(d) || 0;
        const s = storageByDir.get(d);
        const free = (s && s.free_bytes != null) ? ` · ${formatBytes(s.free_bytes)} free` : '';
        chips.push(
          `<button class="mm-drive-chip" data-drive-chip="${escapeHtml(d)}" `
          + `title="Show only models on ${escapeHtml(label)}">`
          + `${escapeHtml(label)} · ${escapeHtml(formatCount(n, 'model'))}${escapeHtml(free)}</button>`,
        );
      }
      const bar = document.createElement('div');
      bar.className = 'mm-drive-filter-bar';
      bar.innerHTML = chips.join('');
      bar.querySelectorAll('[data-drive-chip]').forEach((chip) => {
        chip.addEventListener('click', () => {
          const v = chip.dataset.driveChip;
          setDriveFilter(v === '__all__' ? null : v);
        });
      });
      list.appendChild(bar);
    }

    const recentNames = recentManagedModelNames(models);
    const recentSet = new Set(recentNames);
    const recentEntries = recentNames
      .map((name) => entryByName.get(name))
      .filter(Boolean);

    if (recentEntries.length) {
      const recentEl = document.createElement('section');
      recentEl.className = 'mm-model-group mm-model-group-recent';
      recentEl.innerHTML = `
        <div class="mm-model-group-header">
          <div>
            <div class="mm-model-group-title">Recently Used</div>
            <div class="mm-model-group-copy">${escapeHtml(formatCount(recentEntries.length, 'model'))}</div>
          </div>
        </div>
        <div class="mm-model-group-list"></div>
      `;
      const recentList = recentEl.querySelector('.mm-model-group-list');
      recentEntries.forEach((entry) => {
        recentList.appendChild(entry.card);
      });
      list.appendChild(recentEl);
    }

    for (const backend of ['engine', 'ollama', 'llamacpp']) {
      const entries = groups[backend].filter((entry) => !recentSet.has(entry.model.name));
      if (!entries.length) continue;

      const profile = getBackendProfile(backend);
      const loadedCount = inventory.loadedCounts[backend] || 0;
      const groupEl = document.createElement('section');
      groupEl.className = 'mm-model-group';
      groupEl.dataset.mmFormat = 'gguf';
      groupEl.innerHTML = `
        <div class="mm-model-group-header">
          <div>
            <div class="mm-model-group-title">${escapeHtml(profile.label)}</div>
            <div class="mm-model-group-copy">${escapeHtml(formatCount(entries.length, 'model'))}${loadedCount > 0 ? ` - ${escapeHtml(formatCount(loadedCount, 'ready model', 'ready models'))}` : ''}</div>
          </div>
          <button class="btn btn-sm mm-group-select-btn" data-select-backend="${escapeHtml(backend)}">Add here</button>
        </div>
        <div class="mm-model-group-list"></div>
      `;
      const groupList = groupEl.querySelector('.mm-model-group-list');
      sortModelsForDisplay(entries).forEach((entry) => {
        groupList.appendChild(entry.card);
      });
      groupEl.querySelector('.mm-group-select-btn')?.addEventListener('click', () => {
        showDiscoverPane(backend);
      });
      list.appendChild(groupEl);
    }

    // Re-apply any active drive filter to the freshly-rendered cards.
    applyDriveFilter();

    // Safetensors library — always shown when repos exist on disk, whether or
    // not the vLLM engine is installed (the files are manageable regardless;
    // serving needs the engine, which the group copy notes when it's absent).
    await renderSafetensorsLibrary(list).catch(() => {});
    // Format filter (GGUF / Safetensors) — only when both are present.
    maybeAddFormatFilter(list);
  } catch (err) {
    console.error('[models] refreshModelList failed:', err);
    modalState.inventory = null;
    modalState.managedModels = [];
    modalState.modelEntries = emptyModelEntries();
    renderDeviceGrid();
    renderRoleSettings();
    renderOverview();
    list.innerHTML = '<div class="mm-empty">Could not load models. Is the backend running?</div>';
  }
}

// Per-model sampling editor. Loads the model's override/recommended/effective
// profile, opens the shared design-system editor, and PUTs the override.
// Scope = "all chats using this model" (per-user).
async function openSamplingSheet(modelName) {
  let data;
  try {
    const resp = await fetch(`/api/models/${encodeURIComponent(modelName)}/sampling`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (err) {
    showToast('Could not load sampling profile', 'error');
    console.error('[models] sampling fetch failed:', err);
    return;
  }
  openSamplingEditor({
    scopeLabel: modelName,
    helpText: 'Applies to every chat using this model. Blank = inherit the family default.',
    values: data.override || {},
    effective: data.effective || {},
    recommended: data.recommended || {},
    // Hide knobs this model's backend won't honor (null = show all).
    supported: Array.isArray(data.supported) ? data.supported : null,
    onSave: async (override) => {
      const resp = await fetch(`/api/models/${encodeURIComponent(modelName)}/sampling`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(override),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      showToast('Sampling saved', 'success');
    },
  });
}

function createModelCard(model, isLoaded, isOllama, routerStatus, runningInfo, isEngine = false, engineLoaded = false, engineState = 'idle', driveDir = null) {
  const isSelected = (window.app?.state?.currentModel === model.name);
  const card = document.createElement('div');
  card.className = 'mm-model-card' + (isSelected ? ' mm-selected' : '');
  card.dataset.name = model.name;
  if (driveDir) card.dataset.drive = driveDir;

  const sizeStr = model.size ? formatBytes(model.size) : '';
  const details = model.details || {};
  const backend = details.augmentum_backend || 'ollama';
  const isLlamacpp = backend === 'llamacpp';
  const savedEngineProfile = isEngine ? getEngineModelLoadProfile(model.name) : null;
  const activeEngineStatus = isEngine && modalState.engineStatus?.model_id === model.name
    ? modalState.engineStatus
    : null;
  const metaParts = [];
  if (details.quantization_level) metaParts.push(details.quantization_level);
  if (details.parameter_size) metaParts.push(details.parameter_size);
  else if (details.family) metaParts.push(details.family);
  if (sizeStr) metaParts.push(sizeStr);
  if (runningInfo) {
    if (runningInfo.size_vram) metaParts.push(`VRAM: ${formatBytes(runningInfo.size_vram)}`);
    const ram = (runningInfo.size || 0) - (runningInfo.size_vram || 0);
    if (ram > 0) metaParts.push(`RAM: ${formatBytes(ram)}`);
  }
  if (activeEngineStatus?.load_config?.draft_model) {
    metaParts.push(`Draft: ${formatEngineModelRef(activeEngineStatus.load_config.draft_model)}`);
  }
  // MTP-capable GGUFs advertise built-in next-N predict heads. The
  // engine uses them via --spec-type draft-mtp when engine_mtp_enabled
  // is on — and they take precedence over an external draft model.
  if (model.mtp) metaParts.push('MTP-capable');
  // Filename-collision hint: when two same-named GGUFs exist across
  // overlapping scan dirs, list_models picks one and records the others
  // in details.shadowed_paths so the operator notices the duplicate
  // taking disk space.
  const shadowed = Array.isArray(details.shadowed_paths) ? details.shadowed_paths : [];
  if (shadowed.length) {
    metaParts.push(`Duplicate copy hidden: ${shadowed[0]}${shadowed.length > 1 ? ` (+${shadowed.length - 1} more)` : ''}`);
  }
  if (savedEngineProfile) metaParts.push(`Default: ${engineProfileSummary(savedEngineProfile)}`);

  // Status dot — supports router mode states + engine states
  let statusClass = isLoaded ? 'loaded' : 'unloaded';
  let statusTitle = isLoaded ? 'Loaded in memory' : 'Not loaded';
  if (routerStatus) {
    statusClass = routerStatus;
    statusTitle = routerStatus.charAt(0).toUpperCase() + routerStatus.slice(1);
  }
  if (isEngine) {
    if (engineLoaded) {
      statusClass = 'loaded';
      statusTitle = 'Loaded (engine)';
    } else {
      statusClass = 'unloaded';
      statusTitle = 'Not loaded';
    }
  }

  const isDmr = getCapabilities().is_docker_model_runner;
  let actionsHtml;
  if (isOllama && !isDmr) {
    actionsHtml = `
      ${isLoaded
        ? '<button class="btn btn-sm mm-unload-btn">Stop</button>'
        : '<button class="btn btn-sm mm-load-btn">Use now</button>'
      }
      <button class="btn btn-danger btn-sm mm-delete-btn">Remove</button>
    `;
  } else if (isOllama && isDmr) {
    // DMR: show model list but no load/unload/delete (use docker CLI)
    actionsHtml = '<span class="mm-backend-badge docker">Docker</span>';
  } else if (isLlamacpp && routerStatus) {
    // Router mode — show load/unload buttons
    const isModelLoaded = routerStatus === 'loaded';
    actionsHtml = `
      <span class="mm-backend-badge llamacpp">llama.cpp</span>
      ${isModelLoaded
        ? '<button class="btn btn-sm mm-lcpp-unload-btn">Stop</button>'
        : '<button class="btn btn-sm mm-lcpp-load-btn">Use now</button>'
      }
    `;
  } else if (isEngine) {
    // "Pair vision" — opens a modal that lists every mmproj candidate
    // across all configured model dirs with per-candidate dim-check
    // verdicts, then writes a per-model sidecar JSON on confirm. The
    // button label flips between "Pair vision" and "Vision paired" so
    // the operator sees current state without needing the eye badge.
    const visionLabel = (model.supports_vision || model.vision) ? 'Vision paired' : 'Pair vision';
    const visionClass = (model.supports_vision || model.vision) ? 'mm-pair-vision-btn paired' : 'mm-pair-vision-btn';
    // Slot B action — only when the second resident slot is enabled.
    // Reflects whether THIS model is the one currently in Slot B.
    const isInSlotB = modalState.secondaryEnabled
      && modalState.secondaryModel
      && modalState.secondaryModel === model.name;
    const slotBBtn = !modalState.secondaryEnabled
      ? ''
      : (isInSlotB
        ? '<button class="btn btn-sm mm-engine-slotb-stop-btn" title="Unload this model from the second resident slot">Stop 2nd</button>'
        : '<button class="btn btn-sm mm-engine-slotb-load-btn" title="Load as a second resident model, kept alongside the primary">2nd slot</button>');
    // Slot C action — only when the managed classifier slot is enabled.
    // Reflects whether THIS model is the one currently in Slot C.
    const isInSlotC = modalState.classifierSlotEnabled
      && modalState.classifierSlotModel
      && modalState.classifierSlotModel === model.name;
    const slotCBtn = !modalState.classifierSlotEnabled
      ? ''
      : (isInSlotC
        ? '<button class="btn btn-sm mm-engine-slotc-stop-btn" title="Unload this model from the classifier slot (roles fall back to primary)">Stop clf</button>'
        : '<button class="btn btn-sm mm-engine-slotc-load-btn" title="Load into the classifier slot — serves classifier/utility (and vision if VL); swaps with no container recreate">Classifier</button>');
    actionsHtml = `
      <span class="mm-backend-badge engine">Engine</span>
      ${isInSlotB ? '<span class="mm-backend-badge engine" title="Resident in the second slot">Slot B</span>' : ''}
      ${isInSlotC ? '<span class="mm-backend-badge engine" title="Resident in the classifier slot">Slot C</span>' : ''}
      <button class="btn btn-sm mm-engine-config-btn">Load setup</button>
      ${engineLoaded
        ? '<button class="btn btn-sm mm-engine-unload-btn">Stop</button>'
        : '<button class="btn btn-sm mm-engine-load-btn">Use now</button>'
      }
      ${slotBBtn}
      ${slotCBtn}
      <button class="btn btn-sm ${visionClass}" title="Pair a vision projector (mmproj GGUF) with this model">${visionLabel}</button>
      <button class="btn btn-sm mm-delete-file-btn" title="Delete the GGUF file from disk">Delete</button>
    `;
  } else {
    actionsHtml = `<span class="mm-backend-badge ${escapeHtml(backend)}">${escapeHtml(backend)}</span>`;
  }

  const selectedBadge = isSelected ? '<span class="mm-selected-badge">Current</span>' : '';

  // Drive badge \u2014 which mounted folder this model lives on. Clickable: filters
  // the list to that drive. Only shown when models span multiple drives.
  const driveBadge = (driveDir && hasMultipleDrives())
    ? `<button class="mm-drive-badge" data-drive-filter="${escapeHtml(driveDir)}" `
      + `title="Show only models on ${escapeHtml(driveLabelFromDir(driveDir))}">`
      + `${escapeHtml(driveLabelFromDir(driveDir))}</button>`
    : '';

  card.innerHTML = `
    <div class="mm-status-dot ${statusClass}" title="${escapeHtml(statusTitle)}"></div>
    ${selectedBadge}
    <div class="mm-model-info">
      <div class="mm-model-name"><span class="mm-model-name-label">${escapeHtml(model.name)}</span>${driveBadge}</div>
      <div class="mm-model-meta">${escapeHtml(metaParts.join(' \u00b7 '))}</div>
    </div>
    <div class="mm-model-actions">
      ${actionsHtml}
    </div>
  `;

  // Per-model sampling ("Tuning") — opens an editor for this model's
  // temperature/top_p/top_k/… profile. Available on every backend that
  // serves chat (ollama / llama.cpp / engine), skipped for Docker Model
  // Runner (sampling lives in the docker CLI there). Appended generically
  // so every action branch gets it without touching each one.
  if (!isDmr && (isOllama || isLlamacpp || isEngine)) {
    const actionsEl = card.querySelector('.mm-model-actions');
    if (actionsEl) {
      const tuneBtn = document.createElement('button');
      tuneBtn.className = 'btn btn-sm mm-sampling-btn';
      tuneBtn.title = 'Edit sampling (temperature, top_p, …) for this model';
      tuneBtn.textContent = 'Tuning';
      tuneBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        openSamplingSheet(model.name);
      });
      actionsEl.appendChild(tuneBtn);
    }
  }

  // Drive badge → click-to-filter (stopPropagation so it doesn't trigger
  // any card-level selection).
  const driveBadgeEl = card.querySelector('.mm-drive-badge');
  if (driveBadgeEl) {
    driveBadgeEl.addEventListener('click', (event) => {
      event.stopPropagation();
      setDriveFilter(driveBadgeEl.dataset.driveFilter || null);
    });
  }

  // Ollama actions
  if (isOllama) {
    const loadBtn = card.querySelector('.mm-load-btn');
    const unloadBtn = card.querySelector('.mm-unload-btn');
    const deleteBtn = card.querySelector('.mm-delete-btn');
    if (loadBtn) loadBtn.addEventListener('click', () => loadModel(model.name));
    if (unloadBtn) unloadBtn.addEventListener('click', () => unloadModel(model.name));
    if (deleteBtn) deleteBtn.addEventListener('click', () => {
      showDeleteConfirm(model.name, card);
    });
  }

  // llama.cpp router mode actions
  if (isLlamacpp && routerStatus) {
    const lcppLoadBtn = card.querySelector('.mm-lcpp-load-btn');
    const lcppUnloadBtn = card.querySelector('.mm-lcpp-unload-btn');
    if (lcppLoadBtn) lcppLoadBtn.addEventListener('click', () => lcppLoadModel(model.name));
    if (lcppUnloadBtn) lcppUnloadBtn.addEventListener('click', () => lcppUnloadModel(model.name));
  }

  // Engine v2 actions
  if (isEngine) {
    const engineConfigBtn = card.querySelector('.mm-engine-config-btn');
    const engineLoadBtn = card.querySelector('.mm-engine-load-btn');
    const engineUnloadBtn = card.querySelector('.mm-engine-unload-btn');
    if (engineConfigBtn) {
      engineConfigBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        openEngineLoadSheet(model.name, { source: 'manager' });
      });
    }
    if (engineLoadBtn) {
      engineLoadBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        engineLoadBtn.disabled = true;
        engineLoadBtn.textContent = 'Loading...';
        await loadEngineModelFromManager(model.name, { source: 'manager' });
        engineLoadBtn.disabled = false;
        engineLoadBtn.textContent = 'Use now';
      });
    }
    if (engineUnloadBtn) {
      engineUnloadBtn.addEventListener('click', async () => {
        engineUnloadBtn.disabled = true;
        try {
          const resp = await fetch('/api/engine/v2/models/unload', {method: 'POST'});
          if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || `Unload failed (HTTP ${resp.status})`);
          }
          showToast('Model unloaded', 'success');
        } catch (err) {
          showToast(err.message || 'Unload failed', 'error');
        }
        engineUnloadBtn.disabled = false;
        refreshModelList();
        refreshEngineDashboard();
      });
    }
    const slotbLoadBtn = card.querySelector('.mm-engine-slotb-load-btn');
    if (slotbLoadBtn) {
      slotbLoadBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        slotbLoadBtn.disabled = true;
        slotbLoadBtn.textContent = 'Loading…';
        await loadEngineModelIntoSlotB(model.name);
        slotbLoadBtn.disabled = false;
        slotbLoadBtn.textContent = '2nd slot';
      });
    }
    const slotbStopBtn = card.querySelector('.mm-engine-slotb-stop-btn');
    if (slotbStopBtn) {
      slotbStopBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        await unloadSlotB();
      });
    }
    const slotcLoadBtn = card.querySelector('.mm-engine-slotc-load-btn');
    if (slotcLoadBtn) {
      slotcLoadBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        slotcLoadBtn.disabled = true;
        slotcLoadBtn.textContent = 'Loading…';
        await loadEngineModelIntoSlotC(model.name);
        slotcLoadBtn.disabled = false;
        slotcLoadBtn.textContent = 'Classifier';
      });
    }
    const slotcStopBtn = card.querySelector('.mm-engine-slotc-stop-btn');
    if (slotcStopBtn) {
      slotcStopBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        await unloadSlotC();
      });
    }
    const deleteFileBtn = card.querySelector('.mm-delete-file-btn');
    if (deleteFileBtn) {
      deleteFileBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        deleteLocalModelFile(model, card);
      });
    }
    const pairVisionBtn = card.querySelector('.mm-pair-vision-btn');
    if (pairVisionBtn) {
      pairVisionBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        openProjectorPairer(model.name);
      });
    }
  }

  // Click model info to toggle detail panel
  card.querySelector('.mm-model-info')?.addEventListener('click', async () => {
    let detail = card.querySelector('.mm-model-detail');
    if (detail) { detail.remove(); return; }

    detail = document.createElement('div');
    detail.className = 'mm-model-detail';
    detail.innerHTML = '<div class="mm-detail-loading">Loading...</div>';
    card.appendChild(detail);

    try {
      const resp = await fetch(`/api/models/${encodeURIComponent(model.name)}/info`);
      if (!resp.ok) throw new Error('Failed');
      const info = await resp.json();

      const rows = [];
      if (info.details?.family) rows.push(['Family', info.details.family]);
      if (info.details?.parameter_size) rows.push(['Parameters', info.details.parameter_size]);
      if (info.details?.quantization_level) rows.push(['Quantization', info.details.quantization_level]);
      if (info.context_length) rows.push(['Context', `${info.context_length.toLocaleString()} tokens`]);
      if (info.details?.format) rows.push(['Format', info.details.format]);
      if (isEngine && savedEngineProfile) rows.push(['Saved Load', engineProfileSummary(savedEngineProfile)]);
      if (activeEngineStatus?.load_config?.ctx_size) {
        rows.push(['Loaded At', `${Number(activeEngineStatus.load_config.ctx_size).toLocaleString()} tokens`]);
        if (activeEngineStatus.load_config.draft_model) {
          const draftBits = [formatEngineModelRef(activeEngineStatus.load_config.draft_model)];
          if (activeEngineStatus.load_config.draft_max) {
            draftBits.push(`${Number(activeEngineStatus.load_config.draft_max)} ahead`);
          }
          rows.push(['Spec Decode', draftBits.join(' · ')]);
        }
      }

      detail.innerHTML = rows.length > 0
        ? rows.map(([k, v]) => `<div class="mm-detail-row"><span class="mm-detail-key">${escapeHtml(k)}</span><span class="mm-detail-val">${escapeHtml(String(v))}</span></div>`).join('')
        : '<div class="mm-detail-row">No details available</div>';
    } catch {
      detail.innerHTML = '<div class="mm-detail-row">Could not load details</div>';
    }
  });

  return card;
}

async function lcppLoadModel(name) {
  try {
    const resp = await fetch('/api/llamacpp/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: name }),
    });
    if (!resp.ok) throw new Error('Load failed');
    await adoptLoadedModel(name);
    showToast(`Loading ${name}...`, 'success');
    refreshModelList();
    refreshLcppDashboard();
  } catch (err) {
    showToast('Failed to load model: ' + err.message, 'error');
  }
}

async function lcppUnloadModel(name) {
  try {
    const resp = await fetch('/api/llamacpp/models/unload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: name }),
    });
    if (!resp.ok) throw new Error('Unload failed');
    showToast(`Unloaded ${name}`, 'success');
    refreshModelList();
    refreshLcppDashboard();
  } catch (err) {
    showToast('Failed to unload model: ' + err.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Delete Confirmation
// ---------------------------------------------------------------------------

function showDeleteConfirm(modelName, cardEl) {
  const actions = cardEl.querySelector('.mm-model-actions');

  actions.innerHTML = `
    <div class="mm-delete-confirm">
      <span>Remove ${escapeHtml(modelName)} from this device?</span>
      <button class="btn btn-danger btn-sm mm-confirm-yes">Remove</button>
      <button class="btn btn-sm mm-confirm-no">Cancel</button>
    </div>
  `;

  actions.querySelector('.mm-confirm-yes').addEventListener('click', async () => {
    await deleteModel(modelName);
  });

  actions.querySelector('.mm-confirm-no').addEventListener('click', () => {
    refreshModelList();
  });
}

// ---------------------------------------------------------------------------
// Pull Model (Ollama)
// ---------------------------------------------------------------------------

async function pullModel(name, backendOverride, modelDir) {
  if (!name || !name.trim()) return;
  name = name.trim();

  const backend = backendOverride || q('mm-backend-select').value || 'ollama';

  // Engine downloads route through the gguf_download job system (resumable,
  // survives disconnect). Engine names use the `repo:filename` form.
  if (backend === 'engine') {
    let repo = name;
    let filename = '';
    const colon = name.lastIndexOf(':');
    if (colon > 0 && name.slice(colon + 1).toLowerCase().endsWith('.gguf')) {
      repo = name.slice(0, colon);
      filename = name.slice(colon + 1);
    }
    const body = { name: repo, backend: 'engine' };
    if (filename) body.filename = filename;
    if (modelDir) body.model_dir = modelDir;
    await enqueueGgufDownload(body, filename || repo);
    return;
  }

  // Ollama: keep the inline-streaming flow — Ollama already manages its own
  // resume on its server, and most Ollama pulls are quick.
  const body = { name, backend, stream: true };
  if (modelDir) body.model_dir = modelDir;

  showProgress(name);
  q('mm-pull-btn').disabled = true;
  pullAbortController = new AbortController();

  try {
    const response = await fetch('/api/models/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: pullAbortController.signal,
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || errBody.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    await readNdjsonStream(response, updatePullProgress);

    q('mm-progress-fill').style.width = '100%';
    q('mm-progress-status').textContent = 'Download complete!';
    setTimeout(() => {
      hideProgress();
      refreshModelList();
      fetchModels();
    }, 1500);
  } catch (err) {
    if (err.name === 'AbortError') {
      q('mm-progress-status').textContent = 'Download cancelled.';
      setTimeout(hideProgress, 1000);
    } else {
      q('mm-progress-status').textContent = 'Error: ' + err.message;
      q('mm-progress-fill').style.width = '0%';
    }
  }

  pullAbortController = null;
  q('mm-pull-btn').disabled = false;
}

function updatePullProgress(data) {
  const status = data.status || '';
  q('mm-progress-status').textContent = status;

  if (data.total && data.completed !== undefined) {
    const pct = Math.round((data.completed / data.total) * 100);
    setDeterminateProgress(pct);
    q('mm-progress-status').textContent = `${status} \u2014 ${formatBytes(data.completed)} / ${formatBytes(data.total)} (${pct}%)`;
  } else if (status === 'success') {
    setDeterminateProgress(100);
  } else if (status) {
    setIndeterminateProgress();
  }
}

// ---------------------------------------------------------------------------
// Pull GGUF Model (llama.cpp)
// ---------------------------------------------------------------------------

async function pullGgufModel(repoId, filename, modelDir) {
  // Enqueue as a background job; UI tracks via the Active Downloads panel
  // so the download survives page reload / modal close.
  const body = { backend: 'llamacpp', name: repoId, filename };
  if (modelDir) body.model_dir = modelDir;
  await enqueueGgufDownload(body, filename);
}

// ---------------------------------------------------------------------------
// Active Downloads — server-backed background jobs (gguf_download).
//
// The user-facing pull endpoint enqueues a job and returns immediately. The
// job runs in the background even if the user closes the modal or refreshes
// the page, so downloads no longer disappear when the browser disconnects.
//
// activeDownloadStreams holds one EventSource per visible card so progress
// re-attaches across modal open/close. closeActiveDownloadStreams() tears them
// down on close — the server keeps running, the page just stops listening.
// ---------------------------------------------------------------------------

const activeDownloadStreams = new Map();   // job_id -> EventSource

// Per-job speed tracker for the active-downloads cards. Updated on every
// SSE progress event; reset on completion/failure/disconnect. Entirely
// client-side — derived from `completed` deltas in payloads we already
// receive, so we don't add any backend or DB load. EMA dampens the jitter
// from chunky 4 MB write boundaries so the displayed MB/s doesn't bounce
// every refresh.
const downloadSpeedSamples = new Map();   // job_id -> {bytes, ts, ema}
const _SPEED_EMA_ALPHA = 0.3;             // ~3-sample window at 0.5s cadence
// Faster than any realistic single-pipe network or sustained disk write.
// Anything above this is a discontinuity (resume bump after sidecar
// pre-scan, stream re-attach jumping to mid-download progress, etc.) —
// treat as a fresh baseline rather than feed multi-GB/s into the EMA.
const _SPEED_MAX_REALISTIC_BPS = 2 * 1024 * 1024 * 1024;

function _updateDownloadSpeed(jobId, completed) {
  const now = performance.now();
  const sample = downloadSpeedSamples.get(jobId);
  if (!sample) {
    downloadSpeedSamples.set(jobId, { bytes: completed, ts: now, ema: 0 });
    return 0;
  }
  const dtSec = (now - sample.ts) / 1000;
  // Skip degenerate intervals (back-to-back events, clock skew, byte regress).
  if (dtSec < 0.05 || completed < sample.bytes) {
    return sample.ema;
  }
  const instant = (completed - sample.bytes) / dtSec;
  // Discontinuity guard: a resume after a partial download would spike
  // here if the SSE happens to catch both the pre-scan "completed=0" and
  // the post-scan "completed=resumed_total" emits. Only count what's been
  // newly downloaded since this tracker started.
  if (instant > _SPEED_MAX_REALISTIC_BPS) {
    downloadSpeedSamples.set(jobId, { bytes: completed, ts: now, ema: 0 });
    return 0;
  }
  const ema = sample.ema > 0
    ? (sample.ema * (1 - _SPEED_EMA_ALPHA)) + (instant * _SPEED_EMA_ALPHA)
    : instant;
  downloadSpeedSamples.set(jobId, { bytes: completed, ts: now, ema });
  return ema;
}

function _formatRemaining(secondsRemaining) {
  if (!isFinite(secondsRemaining) || secondsRemaining <= 0) return '';
  const s = Math.round(secondsRemaining);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function _downloadStateMeta(status) {
  switch (status) {
    case 'pending':   return { label: 'Queued',      tone: 'var(--text-muted)' };
    case 'running':   return { label: 'Downloading', tone: 'var(--text-link, var(--accent))' };
    case 'completed': return { label: 'Done',        tone: 'var(--success)' };
    case 'failed':    return { label: 'Failed',      tone: 'var(--danger, #c44)' };
    case 'cancelled': return { label: 'Cancelled',   tone: 'var(--text-muted)' };
    default:          return { label: status || '?', tone: 'var(--text-muted)' };
  }
}

function _downloadActionButtons(d) {
  const isActive = d.status === 'pending' || d.status === 'running';
  if (isActive) {
    return '<button class="btn btn-sm mm-dl-cancel" title="Cancel download">Cancel</button>';
  }

  const buttons = [];
  if (d.status === 'failed' || d.status === 'cancelled') {
    buttons.push('<button class="btn btn-sm mm-dl-retry" title="Retry this download using the saved partial file when possible">Retry</button>');
  }
  if (d.has_partial) {
    buttons.push('<button class="btn btn-sm mm-dl-discard" title="Delete the resumable .part file and remove this entry from activity">Discard partial</button>');
  } else {
    buttons.push('<button class="btn btn-sm mm-dl-clear" title="Remove this entry from activity">Clear</button>');
  }
  return buttons.join('');
}

function _downloadAuxLine(d) {
  if (!d.has_partial) return '';
  const label = d.partial_size > 0
    ? `Resume data saved (${formatBytes(d.partial_size)})`
    : 'Resume data saved';
  return `<div style="font-size:var(--text-xs);color:var(--text-muted);margin-top:4px">${escapeHtml(label)}</div>`;
}

function _wireDownloadCardActions(card, d) {
  const cancelBtn = card.querySelector('.mm-dl-cancel');
  if (cancelBtn) cancelBtn.addEventListener('click', () => cancelDownload(d.job_id));

  const retryBtn = card.querySelector('.mm-dl-retry');
  if (retryBtn) retryBtn.addEventListener('click', () => retryDownload(d));

  const discardBtn = card.querySelector('.mm-dl-discard');
  if (discardBtn) discardBtn.addEventListener('click', () => discardDownload(d));

  const clearBtn = card.querySelector('.mm-dl-clear');
  if (clearBtn) clearBtn.addEventListener('click', () => clearDownload(d.job_id));
}

function _renderDownloadCard(d) {
  const pct = d.total > 0
    ? Math.max(0, Math.min(100, Math.round(d.progress * 100)))
    : (d.status === 'completed' ? 100 : 0);
  const sizeLabel = d.total > 0
    ? `${formatBytes(d.completed)} / ${formatBytes(d.total)}`
    : (d.completed > 0 ? formatBytes(d.completed) : '');
  const meta = _downloadStateMeta(d.status);
  const isActive = d.status === 'pending' || d.status === 'running';
  const indeterminate = d.status === 'pending' || (isActive && d.total <= 0);

  // Speed + ETA only when actively downloading and we have a smoothed
  // sample to show. Hidden during 'pending' (no bytes yet) and on terminal
  // states (would be misleading). Speed is computed in the SSE handler
  // and stashed on the payload — see attachDownloadStream.
  const speedBps = Number(d.speed_bytes_per_sec || 0);
  const showRate = d.status === 'running' && speedBps > 0 && d.total > 0;
  const etaSec = showRate ? (d.total - d.completed) / speedBps : 0;
  const rateLabel = showRate ? `${formatBytes(speedBps)}/s` : '';
  const etaLabel = showRate ? _formatRemaining(etaSec) : '';
  const rateAndEta = showRate
    ? ` &middot; ${escapeHtml(rateLabel)}${etaLabel ? ` &middot; ${escapeHtml(etaLabel)} left` : ''}`
    : '';

  const repoLine = d.repo_id
    ? `<span style="color:var(--text-muted)">${escapeHtml(d.repo_id)}</span>`
    : '';
  const errLine = d.status === 'failed' && d.error
    ? `<div style="color:var(--danger,#c44);font-size:var(--text-xs);margin-top:2px">${escapeHtml(d.error)}</div>`
    : '';

  const card = document.createElement('div');
  card.className = 'mm-download-card';
  card.dataset.jobId = d.job_id;
  card.dataset.status = d.status;
  card.style.cssText = 'border:1px solid var(--border-light);border-radius:var(--radius-md);padding:var(--space-sm);background:var(--surface,transparent)';

  card.innerHTML = `
    <div style="display:flex;align-items:flex-start;gap:var(--space-sm)">
      <div style="flex:1;min-width:0">
        <div style="display:flex;align-items:baseline;gap:var(--space-xs);font-size:var(--text-sm);font-weight:500">
          <span class="mm-dl-name" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(d.filename || d.repo_id || '(unknown)')}</span>
          <span class="mm-backend-badge ${escapeHtml(d.backend)}" style="font-size:10px;padding:1px 6px">${escapeHtml(d.backend || '')}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:var(--text-xs);margin-top:2px;gap:var(--space-sm)">
          ${repoLine}
          <span class="mm-dl-state" style="color:${meta.tone};font-weight:500">${escapeHtml(meta.label)}</span>
        </div>
      </div>
      <div class="mm-dl-actions" style="display:flex;flex-wrap:wrap;justify-content:flex-end;gap:var(--space-xs)">
        ${_downloadActionButtons(d)}
      </div>
    </div>
    <div class="mm-dl-progress" style="margin-top:var(--space-xs);height:6px;background:var(--border-light);border-radius:3px;overflow:hidden;position:relative">
      <div class="mm-dl-progress-fill" style="height:100%;background:${meta.tone};width:${indeterminate ? '40%' : pct + '%'};${indeterminate ? 'animation:mm-indet 1.4s linear infinite' : 'transition:width 200ms ease'}"></div>
    </div>
    <div class="mm-dl-status" style="font-size:var(--text-xs);color:var(--text-muted);margin-top:4px;display:flex;justify-content:space-between;gap:var(--space-sm)">
      <span class="mm-dl-stage" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(d.stage || '')}</span>
      <span class="mm-dl-size">${escapeHtml(sizeLabel)}${d.total > 0 && isActive ? ` (${pct}%)` : ''}${rateAndEta}</span>
    </div>
    ${_downloadAuxLine(d)}
    ${errLine}
  `;

  _wireDownloadCardActions(card, d);
  return card;
}

function _applyDownloadUpdate(card, d) {
  if (!card) return null;
  return _renderDownloadCard(d);
}

async function loadActiveDownloads() {
  const sec = q('mm-downloads-section');
  const list = q('mm-downloads-list');
  const count = q('mm-downloads-count');
  const clearHistoryBtn = q('mm-downloads-clear-history');
  const discardPartialsBtn = q('mm-downloads-discard-partials');
  if (!sec || !list) return;
  let downloads = [];
  try {
    const resp = await fetch('/api/models/downloads?limit=20');
    if (!resp.ok) {
      const d = await resp.json().catch(() => ({}));
      throw new Error(d.detail || d.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    downloads = Array.isArray(data.downloads) ? data.downloads : [];
  } catch (err) {
    // This runs on an 8s poll: when the panel already has rows (live SSE
    // progress), a transient blip must NOT clobber them — leave as-is.
    // But when there's nothing rendered yet, an error must not masquerade
    // as "No download activity yet" (error-vs-empty discipline).
    if (!modalState.downloads?.length) {
      sec.classList.remove('hidden');
      renderFetchError(list, 'download activity', err?.message, loadActiveDownloads);
    }
    return;
  }

  // Hide entirely when nothing to show. Show "x active" badge when active.
  modalState.downloads = downloads;
  const active = downloads.filter((d) => d.status === 'pending' || d.status === 'running');
  // A download that WAS active but no longer is = it finished. Refresh the model
  // list so the newly-downloaded model appears without a manual refresh.
  const nowActiveIds = new Set(active.map((d) => d.id || d.job_id));
  let _anyDownloadFinished = false;
  for (const id of _prevActiveDownloadIds) {
    if (!nowActiveIds.has(id)) { _anyDownloadFinished = true; break; }
  }
  _prevActiveDownloadIds = nowActiveIds;
  if (_anyDownloadFinished) { refreshModelList(); }
  const clearHistoryCount = downloads.filter((d) => (
    d.status === 'completed' || ((d.status === 'failed' || d.status === 'cancelled') && !d.has_partial)
  )).length;
  const discardPartialCount = downloads.filter((d) => (
    (d.status === 'failed' || d.status === 'cancelled') && d.has_partial
  )).length;
  if (downloads.length === 0) {
    sec.classList.remove('hidden');
    list.innerHTML = '<div class="mm-empty">No download activity yet.</div>';
    if (count) {
      count.textContent = '';
      count.style.display = 'none';
    }
    if (clearHistoryBtn) {
      clearHistoryBtn.classList.add('hidden');
      clearHistoryBtn.onclick = null;
    }
    if (discardPartialsBtn) {
      discardPartialsBtn.classList.add('hidden');
      discardPartialsBtn.onclick = null;
    }
    closeActiveDownloadStreams();
    renderOverview();
    return;
  }
  sec.classList.remove('hidden');
  if (count) {
    count.textContent = active.length > 0 ? `${active.length} active` : '';
    count.style.display = active.length > 0 ? '' : 'none';
  }
  if (clearHistoryBtn) {
    clearHistoryBtn.textContent = clearHistoryCount > 0 ? `Clear history (${clearHistoryCount})` : 'Clear history';
    clearHistoryBtn.classList.toggle('hidden', clearHistoryCount === 0);
    clearHistoryBtn.onclick = clearHistoryCount > 0
      ? () => cleanupDownloads({
        statuses: ['completed', 'failed', 'cancelled'],
        requirePartial: false,
        deletePartial: false,
        successMessage: 'Cleared finished activity',
      })
      : null;
  }
  if (discardPartialsBtn) {
    discardPartialsBtn.textContent = discardPartialCount > 0 ? `Discard saved partials (${discardPartialCount})` : 'Discard saved partials';
    discardPartialsBtn.classList.toggle('hidden', discardPartialCount === 0);
    discardPartialsBtn.onclick = discardPartialCount > 0
      ? () => cleanupDownloads({
        statuses: ['failed', 'cancelled'],
        requirePartial: true,
        deletePartial: true,
        successMessage: 'Discarded saved partial downloads',
      })
      : null;
  }

  // Diff-render: keep the existing card if present (so live progress updates
  // don't flash), insert/remove as needed, preserve order.
  const existing = new Map();
  list.querySelectorAll('.mm-download-card').forEach((c) => existing.set(c.dataset.jobId, c));
  const seen = new Set();
  const frag = document.createDocumentFragment();
  for (const d of downloads) {
    seen.add(d.job_id);
    let card = existing.get(d.job_id);
    if (card) {
      card = _applyDownloadUpdate(card, d);
    } else {
      card = _renderDownloadCard(d);
    }
    frag.appendChild(card);
  }
  list.replaceChildren(frag);
  renderOverview();

  // Tear down streams for cards that left the list.
  for (const jobId of Array.from(activeDownloadStreams.keys())) {
    if (!seen.has(jobId)) {
      _detachStream(jobId);
    }
  }
  // Open streams for active downloads that don't have one yet.
  for (const d of active) {
    if (!activeDownloadStreams.has(d.job_id)) {
      attachDownloadStream(d.job_id);
    }
  }
}

function attachDownloadStream(jobId) {
  if (activeDownloadStreams.has(jobId)) return;
  const es = new EventSource(`/api/models/downloads/${encodeURIComponent(jobId)}/stream`);
  activeDownloadStreams.set(jobId, es);
  es.onmessage = (ev) => {
    if (!ev.data) return;
    let payload;
    try { payload = JSON.parse(ev.data); } catch { return; }
    // Compute rolling speed from completed-bytes delta. Stash on the payload
    // so the renderer can show MB/s + ETA without needing any new fetches.
    if ((payload.status === 'pending' || payload.status === 'running') && payload.completed >= 0) {
      payload.speed_bytes_per_sec = _updateDownloadSpeed(jobId, payload.completed);
    }
    const card = q('mm-downloads-list')?.querySelector(`.mm-download-card[data-job-id="${CSS.escape(jobId)}"]`);
    if (card) {
      const next = _applyDownloadUpdate(card, payload);
      if (next && card.parentNode) card.replaceWith(next);
    }
    if (payload.status === 'completed') {
      downloadSpeedSamples.delete(jobId);
      _detachStream(jobId);
      // Refresh list so the new model appears under Installed Models.
      refreshModelList();
      fetchModels();
      // Pull the updated downloads list after a short beat so the completed
      // card transitions out of "active" state in the panel.
      setTimeout(loadActiveDownloads, 600);
    } else if (payload.status === 'failed' || payload.status === 'cancelled') {
      downloadSpeedSamples.delete(jobId);
      _detachStream(jobId);
      setTimeout(loadActiveDownloads, 300);
    }
  };
  es.onerror = () => {
    // Network blip / server restart — close and let the next loadActiveDownloads
    // re-open if the job is still active.
    _detachStream(jobId);
  };
}

function _detachStream(jobId) {
  const es = activeDownloadStreams.get(jobId);
  if (es) {
    try { es.close(); } catch {}
    activeDownloadStreams.delete(jobId);
  }
  // Drop the speed sample too so a re-attach computes a fresh baseline
  // instead of inheriting a stale lastBytes/ts pair across the gap.
  downloadSpeedSamples.delete(jobId);
}

function closeActiveDownloadStreams() {
  for (const jobId of Array.from(activeDownloadStreams.keys())) {
    _detachStream(jobId);
  }
}

async function cancelDownload(jobId) {
  try {
    const resp = await fetch(`/api/models/downloads/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    showToast('Cancel requested', 'info');
    setTimeout(loadActiveDownloads, 200);
  } catch (e) {
    showToast('Failed to cancel: ' + (e.message || e), 'error');
  }
}

async function retryDownloadLegacy(d) {
  // Re-queue the same (repo, filename, dir) — handler resumes from the .part
  // file if it still exists, downloads fresh otherwise.
  if (!d.repo_id || !d.filename) return;
  const body = { backend: d.backend || 'llamacpp', name: d.repo_id, filename: d.filename };
  if (d.model_dir) body.model_dir = d.model_dir;
  await enqueueGgufDownload(body, d.filename);
}

async function retryDownload(d) {
  try {
    const resp = await fetch(`/api/models/downloads/${encodeURIComponent(d.job_id)}/retry`, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${resp.status}`);
    }
    showToast(`Retry queued: ${d.filename || d.repo_id || 'download'}`, 'success');
    await loadActiveDownloads();
  } catch (e) {
    showToast('Failed to retry: ' + (e.message || e), 'error');
  }
}

async function clearDownload(jobId) {
  try {
    const resp = await fetch(`/api/models/downloads/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
    if (resp.status === 404) {
      showToast('That activity item was already gone', 'info');
      await loadActiveDownloads();
      return;
    }
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json().catch(() => ({}));
    showToast(data.already_gone ? 'That activity item was already gone' : 'Removed from activity', data.already_gone ? 'info' : 'success');
    await loadActiveDownloads();
  } catch (e) {
    showToast('Failed to clear: ' + (e.message || e), 'error');
  }
}

async function discardDownload(d) {
  try {
    const resp = await fetch(`/api/models/downloads/${encodeURIComponent(d.job_id)}?delete_partial=true`, {
      method: 'DELETE',
    });
    if (resp.status === 404) {
      showToast('That download activity was already gone', 'info');
      await loadActiveDownloads();
      return;
    }
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json().catch(() => ({}));
    if (data.already_gone) {
      showToast('That download activity was already gone', 'info');
    } else if (data.partial_deleted) {
      showToast(`Discarded partial for ${d.filename || d.repo_id || 'download'}`, 'success');
    } else {
      showToast(`Removed ${d.filename || d.repo_id || 'download'} from activity`, 'success');
    }
    await loadActiveDownloads();
  } catch (e) {
    showToast('Failed to discard partial: ' + (e.message || e), 'error');
  }
}

async function cleanupDownloads({ statuses, requirePartial = null, deletePartial = false, successMessage }) {
  try {
    const resp = await fetch('/api/models/downloads/cleanup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        statuses,
        require_partial: requirePartial,
        delete_partial: deletePartial,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json().catch(() => ({}));
    const removed = Number(data.removed || 0);
    const partialDeleted = Number(data.partial_deleted || 0);
    const errors = Number(data.errors || 0);
    if (!removed && !partialDeleted) {
      showToast('Nothing to clean up right now', 'info');
    } else if (errors > 0) {
      showToast(`${successMessage} (${removed} removed, ${partialDeleted} partials deleted, ${errors} errors)`, 'info');
    } else {
      showToast(successMessage, 'success');
    }
    await loadActiveDownloads();
  } catch (e) {
    showToast('Cleanup failed: ' + (e.message || e), 'error');
  }
}

async function enqueueGgufDownload(body, displayName) {
  try {
    const resp = await fetch('/api/models/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    if (data.existing) {
      showToast(`Already downloading ${displayName || ''}`.trim(), 'info');
    } else {
      showToast(`Download queued: ${displayName || ''}`.trim(), 'success');
    }
    // Keep the picker open so users can queue multiple parts / quants without
    // having to re-search the repo every click.
    await loadActiveDownloads();
  } catch (e) {
    showToast('Failed to start download: ' + (e.message || e), 'error');
  }
}

// ---------------------------------------------------------------------------
// GGUF File Browser
// ---------------------------------------------------------------------------

async function populateDownloadDestinations(backend) {
  const sel = q('mm-gguf-dest');
  if (!sel) return;
  try {
    const resp = await fetch('/api/models/download/destinations');
    if (!resp.ok) { sel.innerHTML = ''; return; }
    const data = await resp.json();
    const dirs = Array.isArray(data.destinations) ? data.destinations : [];
    const preferredDefault = backend === 'engine' ? data.engine_default : data.llamacpp_default;
    sel.innerHTML = dirs
      .map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`)
      .join('') || '<option value="">(no destinations configured)</option>';
    if (preferredDefault && dirs.includes(preferredDefault)) {
      sel.value = preferredDefault;
    }
    // Listener-once: reattach on every populate (cheap; the select is
    // recreated each picker open). The `change` listener swaps the usage
    // line for the newly selected dest; no fetch fires (we already have
    // the cached data from the picker-open fetch below).
    sel.onchange = () => _renderDestinationUsage(sel.value);
    await _refreshDestinationUsage(sel.value);
  } catch (e) {
    sel.innerHTML = '<option value="">(failed to load destinations)</option>';
  }
}

// Cached on the client side (single in-flight fetch + 30s soft TTL) so
// repeated picker opens don't re-hit the server. Server-side cache is
// already 30s, so this is belt-and-suspenders against accidental spam.
let _destinationStorageCache = null;       // {ts, data: [{dir, total_bytes, used_bytes, free_bytes, error?}]}
let _destinationStorageInflight = null;    // Promise dedupe for concurrent calls
const _DEST_STORAGE_TTL_MS = 30_000;

async function _refreshDestinationUsage(currentDir) {
  const usageEl = q('mm-gguf-dest-usage');
  if (!usageEl) return;
  const now = performance.now();
  if (_destinationStorageCache && (now - _destinationStorageCache.ts) < _DEST_STORAGE_TTL_MS) {
    _renderDestinationUsage(currentDir);
    return;
  }
  if (_destinationStorageInflight) {
    await _destinationStorageInflight;
    _renderDestinationUsage(currentDir);
    return;
  }
  _destinationStorageInflight = (async () => {
    try {
      const resp = await fetch('/api/models/storage');
      if (!resp.ok) return;
      const data = await resp.json();
      _destinationStorageCache = {
        ts: performance.now(),
        data: Array.isArray(data.destinations) ? data.destinations : [],
      };
    } catch {
      // Network blip — leave the usage line empty rather than scary-erroring.
    } finally {
      _destinationStorageInflight = null;
    }
  })();
  await _destinationStorageInflight;
  _renderDestinationUsage(currentDir);
}

function _renderDestinationUsage(currentDir) {
  const usageEl = q('mm-gguf-dest-usage');
  if (!usageEl) return;
  if (!_destinationStorageCache || !currentDir) {
    usageEl.innerHTML = '';
    return;
  }
  const entry = _destinationStorageCache.data.find((d) => d.dir === currentDir);
  if (!entry) {
    usageEl.innerHTML = '';
    return;
  }
  if (entry.error) {
    usageEl.innerHTML = `<span style="color:var(--text-muted)">disk usage unavailable: ${escapeHtml(entry.error)}</span>`;
    return;
  }
  const total = Number(entry.total_bytes || 0);
  const free = Number(entry.free_bytes || 0);
  const used = Math.max(0, total - free);
  const pct = total > 0 ? (used / total) * 100 : 0;
  // Bar color hints when free space is getting tight. >90% used = red,
  // >75% used = amber, otherwise neutral. Helps the user notice before
  // they queue a 50 GB download to a near-full drive.
  const tone = pct >= 90 ? 'var(--danger,#c44)'
    : pct >= 75 ? 'var(--warning,#c80)'
    : 'var(--accent)';
  usageEl.innerHTML = `
    <div style="display:flex;justify-content:space-between;gap:var(--space-sm);align-items:baseline">
      <span>${escapeHtml(formatBytes(free))} free of ${escapeHtml(formatBytes(total))}</span>
      <span>${pct.toFixed(0)}% used</span>
    </div>
    <div style="height:4px;background:var(--border-light);border-radius:2px;overflow:hidden;margin-top:2px">
      <div style="height:100%;width:${pct.toFixed(1)}%;background:${tone};transition:width 200ms ease"></div>
    </div>
  `;
}

// Multi-part GGUFs ship as ``-NNNNN-of-NNNNN.gguf``. They MUST load together,
// so we group them in the picker and let one click queue the whole set.
const MULTI_PART_RE = /^(.+?)-(\d{1,5})-of-(\d{1,5})\.gguf$/i;
// Vision-language models pair a base quant with one or more ``mmproj-*.gguf``
// projectors. They're optional, so we surface them as a checkbox per quant.
const MMPROJ_RE = /^mmproj/i;

// Precision preference for the auto-include checkbox. Repos commonly ship
// the same projector at multiple precisions (BF16/F16/F32, sometimes Q8) —
// the runtime only loads one, so we pick the most-compatible default and
// expose the rest as standalone rows for manual grab. F16 wins because it
// runs everywhere; BF16 needs GPU compute capability >= 8.0; F32 is a 2x
// larger file with no visible quality benefit for the projector layers.
const MMPROJ_PRECISION_RANK = {
  f16: 1,
  bf16: 2,
  q8_0: 3,
  q8: 3,
  q5_k: 4,
  q5: 4,
  q4_k: 5,
  q4: 5,
  f32: 6,
};

function _pickPreferredMmproj(mmprojs) {
  if (!mmprojs.length) return null;
  if (mmprojs.length === 1) return mmprojs[0];
  const scored = mmprojs.map((m) => {
    const leaf = (m.filename || '').split('/').pop() || '';
    const tag = leaf.replace(/^mmproj[-_]?/i, '').replace(/\.gguf$/i, '').toLowerCase();
    const rank = MMPROJ_PRECISION_RANK[tag] ?? 99;
    return { file: m, rank };
  });
  scored.sort((a, b) => a.rank - b.rank);
  return scored[0].file;
}

function categorizeGgufFiles(files) {
  const groups = new Map();   // basename -> { basename, parts: [], of, totalSize }
  const mmprojs = [];
  const singles = [];
  for (const f of files) {
    const leaf = (f.filename || '').split('/').pop() || f.filename || '';
    const m = leaf.match(MULTI_PART_RE);
    if (m) {
      const dir = (f.filename || '').slice(0, (f.filename || '').length - leaf.length);
      const basename = dir + m[1];
      const idx = parseInt(m[2], 10);
      const of = parseInt(m[3], 10);
      if (!groups.has(basename)) {
        groups.set(basename, { basename, parts: [], of, totalSize: 0 });
      }
      const g = groups.get(basename);
      g.parts.push({ filename: f.filename, size: f.size || 0, index: idx });
      g.totalSize += (f.size || 0);
    } else if (MMPROJ_RE.test(leaf)) {
      mmprojs.push(f);
    } else {
      singles.push(f);
    }
  }
  for (const g of groups.values()) g.parts.sort((a, b) => a.index - b.index);
  // Stable, predictable order: groups first (largest models tend to ship
  // multi-part), then single quants, then mmprojs at the bottom.
  return {
    groups: Array.from(groups.values()),
    singles,
    mmprojs,
    chosenMmproj: _pickPreferredMmproj(mmprojs),
  };
}

// Download a full safetensors repo + register it with the vLLM engine. Reuses
// the multi-drive destinations picker; the repo lands in a per-model subdir and
// llama-swap auto-serves it (--watch-config).
async function enqueueSafetensors(repoId, filenames, modelDir) {
  if (!filenames || !filenames.length) { showToast('No files to download', 'error'); return; }
  const body = { name: repoId, filenames };
  if (modelDir) body.model_dir = modelDir;
  try {
    const resp = await fetch('/api/models/pull/safetensors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      showToast(`${repoId}: ${data.detail || data.error || 'download failed'}`, 'error');
      return;
    }
    const where = data.serve_path ? ` → ${data.serve_path}` : '';
    const needsEngine = !getCapabilities().has_vllm;
    showToast(
      `${data.model_name || repoId}: downloading${where}${data.existing ? ' (already in flight)' : ''}`
        + (needsEngine ? ' — install the vLLM Engine (Discover) to serve it' : ''),
      data.existing ? 'info' : 'success',
    );
    hideSearchResults();
    await loadActiveDownloads();
  } catch (err) {
    showToast(`${repoId}: ${err.message || 'download failed'}`, 'error');
  }
}

async function populateSafetensorsDestinations() {
  const sel = q('mm-st-dest');
  if (!sel) return;
  try {
    const resp = await fetch('/api/models/download/destinations');
    if (!resp.ok) { sel.innerHTML = '<option value="">(default)</option>'; return; }
    const data = await resp.json();
    const dirs = Array.isArray(data.destinations) ? data.destinations : [];
    sel.innerHTML = dirs.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join('')
      || '<option value="">(default)</option>';
    if (data.engine_default && dirs.includes(data.engine_default)) sel.value = data.engine_default;
  } catch {
    sel.innerHTML = '<option value="">(default)</option>';
  }
}

async function enqueueGgufBundle(repoId, filenames, modelDir, backend, displayName) {
  if (!filenames.length) return;
  // Bundle endpoint enqueues ONE job that downloads all files in parallel.
  // The previous per-file loop POSTed N jobs that queued serially behind
  // the single-worker job runner, so a 4-shard model used to take ~4x as
  // long as a single-shard one even though each shard internally uses 8
  // parallel ranged GETs.
  const body = { backend, name: repoId, filenames };
  if (modelDir) body.model_dir = modelDir;
  let tone = 'success';
  let summary;
  try {
    const resp = await fetch('/api/models/pull/bundle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({}));
      tone = 'error';
      summary = errBody.detail || errBody.error || `HTTP ${resp.status}`;
    } else {
      const data = await resp.json();
      const noun = filenames.length === 1 ? 'file' : `${filenames.length} files`;
      summary = data.existing ? `${noun} already in flight` : `${noun} queued`;
      tone = data.existing ? 'info' : 'success';
    }
  } catch (err) {
    tone = 'error';
    summary = err?.message || 'enqueue failed';
  }
  showToast(`${displayName}: ${summary}`, tone);
  await loadActiveDownloads();
}

// Middle-truncate a filename so the quant tag at the end stays visible at
// every viewport width. End-truncate ellipsis chops "...UD-Q4_K_XL.gguf"
// down to "...UD-Q4_K_..." or worse on half-screen displays, hiding the
// exact info the user is choosing. We split into a truncatable prefix and
// a never-shrunk suffix; the tail (~14 chars) is enough for any quant
// tag plus the .gguf extension.
function _renderTruncatableFilename(text, suffixChars = 14) {
  const escaped = escapeHtml(text);
  if (text.length <= suffixChars + 6) {
    return `<span class="mm-gguf-name mm-gguf-name-short">${escaped}</span>`;
  }
  const splitAt = text.length - suffixChars;
  return `<span class="mm-gguf-name" title="${escaped}"
    ><span class="mm-gguf-name-prefix">${escapeHtml(text.slice(0, splitAt))}</span
    ><span class="mm-gguf-name-suffix">${escapeHtml(text.slice(splitAt))}</span
    ></span>`;
}

function _renderMmprojCheckbox(chosenMmproj, mmprojs) {
  if (!chosenMmproj) return '';
  const leaf = chosenMmproj.filename.split('/').pop() || chosenMmproj.filename;
  const sizeStr = formatModelFileSize(chosenMmproj.size || 0);
  const altCount = Math.max(0, mmprojs.length - 1);
  const altHint = altCount > 0
    ? ` <span style="opacity:0.7" title="Other precisions are listed below \u2014 only one is needed at runtime.">(+${altCount} other precision${altCount === 1 ? '' : 's'} available)</span>`
    : '';
  return `
    <label class="mm-mmproj-toggle" style="display:flex;align-items:center;gap:6px;font-size:var(--text-xs);color:var(--text-muted);margin-top:4px;cursor:pointer">
      <input type="checkbox" class="mm-mmproj-cb" checked>
      <span>Include vision projector (${escapeHtml(leaf)}, ${escapeHtml(sizeStr)})${altHint}</span>
    </label>
  `;
}

function _renderGroupRow(repoId, backend, group, chosenMmproj, mmprojs) {
  const sizeStr = formatModelFileSize(group.totalSize);
  const baseLeaf = group.basename.split('/').pop() || group.basename;
  const partsHtml = group.parts.map((p) => `
    <div class="mm-gguf-subitem" style="display:flex;align-items:center;gap:var(--space-xs);padding:4px 0 4px var(--space-md);font-size:var(--text-xs);color:var(--text-muted)">
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">Part ${p.index} of ${group.of} \u00b7 ${escapeHtml(p.filename)}</span>
      <span>${escapeHtml(formatModelFileSize(p.size))}</span>
    </div>
  `).join('');
  const mmprojCheckbox = _renderMmprojCheckbox(chosenMmproj, mmprojs);
  return `
    <div class="mm-gguf-item mm-gguf-group" data-basename="${escapeHtml(group.basename)}">
      <div class="mm-gguf-row" style="display:flex;align-items:center;gap:var(--space-xs)">
        <button class="mm-gguf-expander" title="Show parts" style="background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:var(--text-xs);padding:2px 4px;width:18px">\u25b6</button>
        ${_renderTruncatableFilename(baseLeaf)}
        <span class="mm-backend-badge" style="font-size:10px;padding:1px 6px;background:color-mix(in srgb, var(--accent) 12%, transparent);color:var(--accent)">${group.parts.length} parts</span>
        <span class="mm-gguf-size">${escapeHtml(sizeStr)}</span>
        <button class="btn btn-primary btn-sm mm-gguf-dl-group">Download all</button>
      </div>
      ${mmprojCheckbox}
      <div class="mm-gguf-subitems hidden" style="margin-top:4px">${partsHtml}</div>
    </div>
  `;
}

function _renderSingleRow(repoId, backend, file, chosenMmproj, mmprojs) {
  const mmprojCheckbox = _renderMmprojCheckbox(chosenMmproj, mmprojs);
  return `
    <div class="mm-gguf-item" data-filename="${escapeHtml(file.filename)}" data-repo="${escapeHtml(repoId)}" data-backend="${escapeHtml(backend)}">
      <div class="mm-gguf-row" style="display:flex;align-items:center;gap:var(--space-xs)">
        ${_renderTruncatableFilename(file.filename)}
        <span class="mm-gguf-size">${escapeHtml(formatModelFileSize(file.size))}</span>
        <button class="btn btn-primary btn-sm mm-gguf-dl-btn">Download</button>
      </div>
      ${mmprojCheckbox}
    </div>
  `;
}

function _renderMmprojRow(repoId, backend, file) {
  const leaf = file.filename.split('/').pop() || file.filename;
  return `
    <div class="mm-gguf-item mm-gguf-mmproj" data-filename="${escapeHtml(file.filename)}" data-repo="${escapeHtml(repoId)}" data-backend="${escapeHtml(backend)}" style="opacity:0.85">
      <div class="mm-gguf-row" style="display:flex;align-items:center;gap:var(--space-xs)">
        ${_renderTruncatableFilename(leaf)}
        <span class="mm-backend-badge" style="font-size:10px;padding:1px 6px;background:color-mix(in srgb, var(--accent) 10%, transparent);color:var(--text-muted)">vision projector</span>
        <span class="mm-gguf-size">${escapeHtml(formatModelFileSize(file.size))}</span>
        <button class="btn btn-sm mm-gguf-dl-btn">Download</button>
      </div>
    </div>
  `;
}

async function fetchGgufFiles(repoId, backend = 'llamacpp', opts = {}) {
  if (!repoId) return;
  const fromSearch = !!opts.fromSearch;

  const picker = q('mm-gguf-picker');
  const list = q('mm-gguf-list');
  picker.classList.remove('hidden');
  // When the picker was reached by clicking a search result, offer a one-click
  // "Back to results" so the user isn't stranded having to clear + re-type the
  // repo name to see the other matches again. Hidden otherwise (typed-repo /
  // chip paths have nothing to go back to).
  syncGgufBackButton(picker, fromSearch);
  list.innerHTML = '<div class="mm-empty">Loading files\u2026</div>';

  populateDownloadDestinations(backend);

  try {
    const resp = await fetch(`/api/models/gguf/list?repo=${encodeURIComponent(repoId)}`);
    const data = await resp.json();

    if (data.error) {
      // Server-reported failure (HF unreachable, bad repo, rate limit) —
      // render as an ERROR with retry, not mm-empty, so it can't be read
      // as "this repo has no files".
      renderFetchError(list, `files for ${repoId}`, data.error,
        () => fetchGgufFiles(repoId, backend, opts));
      return;
    }
    if (!data.files || data.files.length === 0) {
      list.innerHTML = '<div class="mm-empty">No .gguf files found in this repo.</div>';
      return;
    }

    const { groups, singles, mmprojs, chosenMmproj } = categorizeGgufFiles(data.files);
    const parts = [];
    for (const g of groups) parts.push(_renderGroupRow(repoId, backend, g, chosenMmproj, mmprojs));
    for (const s of singles) parts.push(_renderSingleRow(repoId, backend, s, chosenMmproj, mmprojs));
    for (const m of mmprojs) parts.push(_renderMmprojRow(repoId, backend, m));
    list.innerHTML = parts.join('');

    const getModelDir = () => {
      const destSel = q('mm-gguf-dest');
      return destSel && destSel.value ? destSel.value : '';
    };
    // Auto-include only the chosen projector — the runtime loads exactly one
    // and other precisions are still grabbable from their standalone rows.
    const collectMmprojFilenames = (item) => {
      const cb = item.querySelector('.mm-mmproj-cb');
      if (!cb || !cb.checked || !chosenMmproj) return [];
      return [chosenMmproj.filename];
    };

    // Expanders for multi-part groups
    list.querySelectorAll('.mm-gguf-expander').forEach((exp) => {
      exp.addEventListener('click', (e) => {
        e.stopPropagation();
        const subs = exp.closest('.mm-gguf-group')?.querySelector('.mm-gguf-subitems');
        if (!subs) return;
        const open = subs.classList.toggle('hidden');
        exp.textContent = open ? '\u25b6' : '\u25bc';
      });
    });

    // "Download all" for multi-part groups (+ optional mmproj)
    list.querySelectorAll('.mm-gguf-dl-group').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const item = btn.closest('.mm-gguf-group');
        if (!item) return;
        const basename = item.dataset.basename;
        const group = groups.find((g) => g.basename === basename);
        if (!group) return;
        const filenames = group.parts.map((p) => p.filename).concat(collectMmprojFilenames(item));
        btn.disabled = true;
        btn.textContent = 'Queuing...';
        try {
          await enqueueGgufBundle(
            repoId, filenames, getModelDir(), backend,
            (basename.split('/').pop() || basename) + ` (${group.parts.length} parts${filenames.length > group.parts.length ? ' + mmproj' : ''})`,
          );
        } finally {
          btn.disabled = false;
          btn.textContent = 'Download all';
        }
      });
    });

    // Single-quant Download (+ optional mmproj sibling)
    list.querySelectorAll('.mm-gguf-item:not(.mm-gguf-group):not(.mm-gguf-mmproj) .mm-gguf-dl-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const item = btn.closest('.mm-gguf-item');
        const filename = item.dataset.filename;
        const filenames = [filename, ...collectMmprojFilenames(item)];
        btn.disabled = true;
        const original = btn.textContent;
        btn.textContent = 'Queuing...';
        try {
          await enqueueGgufBundle(
            repoId, filenames, getModelDir(), backend,
            filename.split('/').pop() + (filenames.length > 1 ? ' (+ mmproj)' : ''),
          );
        } finally {
          btn.disabled = false;
          btn.textContent = original;
        }
      });
    });

    // Standalone mmproj Download
    list.querySelectorAll('.mm-gguf-mmproj .mm-gguf-dl-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const item = btn.closest('.mm-gguf-item');
        const filename = item.dataset.filename;
        btn.disabled = true;
        const original = btn.textContent;
        btn.textContent = 'Queuing...';
        try {
          await enqueueGgufBundle(repoId, [filename], getModelDir(), backend, filename.split('/').pop() || filename);
        } finally {
          btn.disabled = false;
          btn.textContent = original;
        }
      });
    });
  } catch (err) {
    renderFetchError(list, `files for ${repoId}`, err?.message,
      () => fetchGgufFiles(repoId, backend));
  }
}

// ---------------------------------------------------------------------------
// Delete / Load / Unload
// ---------------------------------------------------------------------------

async function deleteModel(name) {
  try {
    const resp = await fetch('/api/delete', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!resp.ok) throw new Error('Delete failed');
    refreshModelList();
    fetchModels();
  } catch (err) {
    showToast('Failed to delete model: ' + err.message, 'error');
  }
}

// In-card confirm + DELETE for an engine GGUF file. Two-step (confirm then
// commit) so users can't fat-finger a 60GB delete. Resolves the path via
// /info because the model card only carries the friendly name.
function deleteLocalModelFile(model, card) {
  const actions = card.querySelector('.mm-model-actions');
  if (!actions) return;
  const friendly = formatEngineModelRef(model.name) || model.name;
  const original = actions.innerHTML;
  // Deleting a model that's pinned to Slot B / Slot C would leave a
  // dangling pin (the slot keeps serving from the deleted file's mmap
  // until restart, then fails to reload). Surface it in the confirm and
  // unload the slot before deleting.
  const pinnedSlots = [];
  const primaryModel = modalState.engineStatus?.model_id || '';
  if (primaryModel && primaryModel === model.name) {
    pinnedSlots.push({ label: 'the primary engine slot', endpoint: '/api/engine/v2/models/unload' });
  }
  if (modalState.secondaryModel && modalState.secondaryModel === model.name) {
    pinnedSlots.push({ label: 'Slot B', endpoint: '/api/engine/v2/secondary/unload' });
  }
  if (modalState.classifierSlotModel && modalState.classifierSlotModel === model.name) {
    pinnedSlots.push({ label: 'Slot C (classifier)', endpoint: '/api/engine/v2/classifier/unload' });
  }
  const pinWarning = pinnedSlots.length
    ? `<span class="mm-delete-pin-warning">Currently loaded in ${escapeHtml(pinnedSlots.map(p => p.label).join(' and '))} — deleting will unload it first.</span>`
    : '';
  actions.innerHTML = `
    <div class="mm-delete-confirm">
      <span>Delete <strong>${escapeHtml(friendly)}</strong> from disk?</span>
      ${pinWarning}
      <button class="btn btn-danger btn-sm mm-confirm-yes">Delete file</button>
      <button class="btn btn-sm mm-confirm-no">Cancel</button>
    </div>
  `;
  actions.querySelector('.mm-confirm-no').addEventListener('click', () => {
    actions.innerHTML = original;
    refreshModelList();
  });
  actions.querySelector('.mm-confirm-yes').addEventListener('click', async () => {
    const yes = actions.querySelector('.mm-confirm-yes');
    const no = actions.querySelector('.mm-confirm-no');
    if (yes) yes.disabled = true;
    if (no) no.disabled = true;
    try {
      // Unload any slot pins first (warned about in the confirm above) so
      // the delete never leaves Slot B/C pointing at a missing file.
      for (const pin of pinnedSlots) {
        const resp = await fetch(pin.endpoint, { method: 'POST' });
        if (!resp.ok) {
          const d = await resp.json().catch(() => ({}));
          throw new Error(`Couldn't unload ${pin.label} before delete: ${d.detail || `HTTP ${resp.status}`}`);
        }
        showToast(`${pin.label} unloaded`, 'info');
      }
      const record = await resolveEngineModelRecord(model.name).catch(() => null);
      let path = record?.path || '';
      if (!path) {
        const infoResp = await fetch(`/api/models/${encodeURIComponent(model.name)}/info`);
        if (!infoResp.ok) throw new Error(`info lookup failed (HTTP ${infoResp.status})`);
        const info = await infoResp.json();
        path = info.model_path || info.path || model.name;
      }
      const delResp = await fetch('/api/models/local', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });
      if (!delResp.ok) {
        const err = await delResp.json().catch(() => ({}));
        throw new Error(err.detail || err.error || `HTTP ${delResp.status}`);
      }
      const result = await delResp.json();
      showToast(`Deleted ${friendly}${result.size ? ` (${formatBytes(result.size)} freed)` : ''}`, 'success');
      refreshModelList();
      fetchModels();
      refreshEngineDashboard();
    } catch (err) {
      showToast('Delete failed: ' + (err.message || err), 'error');
      actions.innerHTML = original;
    }
  });
}

async function loadModel(name) {
  try {
    const resp = await fetch(`/api/models/${encodeURIComponent(name)}/load`, { method: 'POST' });
    if (!resp.ok) throw new Error('Load failed');
    await adoptLoadedModel(name);
    refreshModelList();
  } catch (err) {
    showToast('Failed to load model: ' + err.message, 'error');
  }
}

async function adoptLoadedModel(name) {
  const value = (name || '').trim();
  if (!value) return;

  if (window.app?.state) window.app.state.currentModel = value;
  if (window.app?.dom?.modelName) window.app.dom.modelName.textContent = value;
  localStorage.setItem('augmentum-selected-model', value);
  addToRecentModels(value);
  pushPrimaryChatModel(value);
  updateThinkingToggleUI(value);

  try {
    await invalidateModelCache('models');
  } catch { /* best effort */ }
  try {
    await fetchModels();
  } catch { /* best effort */ }
}

async function unloadModel(name) {
  try {
    const resp = await fetch(`/api/models/${encodeURIComponent(name)}/unload`, { method: 'POST' });
    if (!resp.ok) throw new Error('Unload failed');
    refreshModelList();
  } catch (err) {
    showToast('Failed to unload model: ' + err.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Progress Helpers
// ---------------------------------------------------------------------------

function showProgress(modelName) {
  q('mm-progress-area').classList.remove('hidden');
  q('mm-progress-model').textContent = modelName;
  q('mm-progress-fill').classList.remove('mm-indeterminate');
  q('mm-progress-fill').style.width = '0%';
  q('mm-progress-status').textContent = 'Preparing...';
}

function hideProgress() {
  q('mm-progress-area').classList.add('hidden');
  q('mm-progress-fill').classList.remove('mm-indeterminate');
  q('mm-progress-fill').style.width = '0%';
  q('mm-progress-status').textContent = '';
}

// ---------------------------------------------------------------------------
// NDJSON Stream Reader
// ---------------------------------------------------------------------------

async function readNdjsonStream(response, onData) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.trim()) continue;
      try { onData(JSON.parse(line)); } catch { /* skip malformed */ }
    }
  }

  if (buffer.trim()) {
    try { onData(JSON.parse(buffer)); } catch { /* skip */ }
  }
}

// ---------------------------------------------------------------------------
// llama.cpp Server Dashboard
// ---------------------------------------------------------------------------

async function refreshLcppDashboard() {
  const caps = getCapabilities();
  const dashboard = q('mm-lcpp-dashboard');
  if (!dashboard) return;

  if (!caps.has_llamacpp) {
    modalState.lcppStatus = null;
    dashboard.classList.remove('active');
    renderDeviceGrid();
    return;
  }
  dashboard.classList.add('active');

  try {
    const resp = await fetch('/api/llamacpp/status');
    if (!resp.ok) throw new Error('status fetch failed');
    const data = await resp.json();
    modalState.lcppStatus = data;

    renderServerLed(data.health);
    renderServerStats(data.props, data.slots, data.is_router_mode);
    renderSlots(data.slots);
    renderLora(data.lora_adapters);
    renderDeviceGrid();
  } catch {
    modalState.lcppStatus = { health: { status: 'unreachable' } };
    renderServerLed({ status: 'unreachable' });
    renderDeviceGrid();
  }
}

function renderServerLed(health) {
  const dot = q('mm-led-dot');
  const label = q('mm-led-label');
  if (!dot || !label) return;

  const status = health?.status || 'unreachable';
  dot.className = 'mm-led-dot';

  if (status === 'ok') {
    dot.classList.add('ok');
    label.textContent = 'ONLINE';
  } else if (status === 'loading model' || status === 'loading') {
    dot.classList.add('loading');
    label.textContent = 'LOADING';
  } else if (status === 'unreachable') {
    label.textContent = 'OFFLINE';
  } else {
    dot.classList.add('error');
    label.textContent = status.toUpperCase();
  }
}

function renderServerStats(props, slots, isRouter) {
  const container = q('mm-lcpp-stats');
  if (!container) return;

  const genSettings = props?.default_generation_settings || {};
  const modelName = genSettings.model || props?.model_path || '—';
  const shortModel = modelName.split('/').pop().replace('.gguf', '');
  const totalSlots = props?.total_slots || slots?.length || 0;
  const busySlots = (slots || []).filter(s => s.is_processing).length;
  const chatTemplate = props?.chat_template || '—';

  const cells = [
    { label: 'Model', value: shortModel },
    { label: 'Slots', value: `${busySlots} / ${totalSlots} busy` },
    { label: 'Context', value: genSettings.n_ctx ? `${genSettings.n_ctx} tok` : '—' },
    { label: 'Mode', value: isRouter ? 'Router' : 'Single' },
  ];

  if (chatTemplate !== '—') {
    cells.push({ label: 'Template', value: chatTemplate });
  }

  container.innerHTML = cells.map(c => `
    <div class="mm-stat-cell">
      <div class="mm-stat-label">${escapeHtml(c.label)}</div>
      <div class="mm-stat-value" title="${escapeHtml(c.value)}">${escapeHtml(c.value)}</div>
    </div>
  `).join('');
}

function renderSlots(slots) {
  const row = q('mm-slots-row');
  const section = q('mm-slots-section');
  if (!row || !section) return;

  if (!slots || slots.length === 0) {
    section.style.display = 'none';
    return;
  }
  section.style.display = '';

  row.innerHTML = slots.map(slot => {
    const id = slot.id ?? '?';
    const busy = slot.is_processing;
    const nCtx = slot.n_ctx || 0;
    const nDecoded = slot.next_token?.n_decoded || 0;
    const usage = nCtx > 0 ? Math.round((nDecoded / nCtx) * 100) : 0;
    const fillClass = usage > 90 ? 'critical' : usage > 70 ? 'high' : '';

    return `
      <div class="mm-slot-block">
        <button class="mm-slot-erase" title="Clear cache" data-slot="${id}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        </button>
        <div class="mm-slot-head">
          <span class="mm-slot-id">Slot ${id}</span>
          <span class="mm-slot-status ${busy ? 'busy' : 'idle'}">${busy ? 'BUSY' : 'IDLE'}</span>
        </div>
        <div class="mm-slot-bar-track">
          <div class="mm-slot-bar-fill ${fillClass}" style="width:${usage}%"></div>
        </div>
        <div class="mm-slot-tokens">${nDecoded} / ${nCtx} tokens</div>
      </div>
    `;
  }).join('');

  // Wire erase buttons
  row.querySelectorAll('.mm-slot-erase').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const slotId = btn.dataset.slot;
      try {
        const resp = await fetch(`/api/llamacpp/slots/${slotId}/erase`, { method: 'POST' });
        if (resp.ok) {
          showToast(`Slot ${slotId} cache cleared`, 'success');
          refreshLcppDashboard();
        } else {
          showToast('Failed to clear slot', 'error');
        }
      } catch {
        showToast('Failed to clear slot', 'error');
      }
    });
  });
}

function renderLora(adapters) {
  const section = q('mm-lora-section');
  const list = q('mm-lora-list');
  if (!section || !list) return;

  if (!adapters || adapters.length === 0) {
    section.classList.add('hidden');
    return;
  }
  section.classList.remove('hidden');

  list.innerHTML = adapters.map(a => {
    const name = a.path ? a.path.split('/').pop() : `Adapter ${a.id}`;
    const scale = typeof a.scale === 'number' ? a.scale : 1.0;
    return `
      <div class="mm-lora-item" data-lora-id="${a.id}">
        <span class="mm-lora-name" title="${escapeHtml(a.path || '')}">${escapeHtml(name)}</span>
        <div class="mm-lora-scale">
          <input type="range" class="mm-lora-slider" min="0" max="1" step="0.05" value="${scale}" data-lora-id="${a.id}">
          <span class="mm-lora-value">${scale.toFixed(2)}</span>
        </div>
      </div>
    `;
  }).join('');

  // Slider live readout
  list.querySelectorAll('.mm-lora-slider').forEach(slider => {
    slider.addEventListener('input', () => {
      const val = slider.closest('.mm-lora-item').querySelector('.mm-lora-value');
      if (val) val.textContent = parseFloat(slider.value).toFixed(2);
    });
  });

  // Apply button
  const applyBtn = q('mm-lora-apply-btn');
  if (applyBtn) {
    applyBtn.onclick = async () => {
      const items = list.querySelectorAll('.mm-lora-item');
      const payload = Array.from(items).map(item => ({
        id: parseInt(item.dataset.loraId, 10),
        scale: parseFloat(item.querySelector('.mm-lora-slider').value),
      }));
      try {
        const resp = await fetch('/api/llamacpp/lora', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ adapters: payload }),
        });
        if (resp.ok) {
          showToast('LoRA scales updated', 'success');
        } else {
          showToast('Failed to update LoRA scales', 'error');
        }
      } catch {
        showToast('Failed to update LoRA scales', 'error');
      }
    };
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export function initModels() {
  // "Models" button in header — wire it up
  const btn = document.getElementById('manage-models-btn');
  if (btn) btn.addEventListener('click', openModelManager);
  if (!modelManagerMediaQuery && typeof window !== 'undefined' && window.matchMedia) {
    modelManagerMediaQuery = window.matchMedia('(max-width: 900px)');
    const onChange = () => updateModelManagerVisibility();
    if (modelManagerMediaQuery.addEventListener) {
      modelManagerMediaQuery.addEventListener('change', onChange);
    } else if (modelManagerMediaQuery.addListener) {
      modelManagerMediaQuery.addListener(onChange);
    }
  }
  window.openAugmentumModelManager = openModelManager;
  window.openAugmentumEngineLoadSheet = openEngineLoadSheet;
  // Exposed on window (not imported) to match how settings.js already reaches
  // the load sheet — the two modules have no direct import edge.
  window.augmentumLoadModelIntoSlot = loadModelIntoSlot;
  updateModelManagerVisibility();
}

/** Show the Models button when a local runtime or the built-in engine is available. */
export function updateModelManagerVisibility() {
  const btn = document.getElementById('manage-models-btn');
  const overflowItem = document.querySelector('.header-overflow-item[data-action="manage-models"]');
  const dropdownManageBtn = document.getElementById('model-dropdown-manage-btn');
  if (btn) btn.style.display = 'none';
  if (overflowItem) overflowItem.style.display = 'none';
  if (dropdownManageBtn) dropdownManageBtn.hidden = false;
}
