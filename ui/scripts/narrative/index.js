/* ==========================================================================
   Augmentum — Narrative Module
   Character grid, narrative state polling, card editor, lorebook
   ========================================================================== */

import { app, escapeHtml, showToast, showChoiceToast, showConfirm } from '../app.js';
import { getSettings } from '../settings.js';
import { scheduleAutosize, resizeNow } from '../utils/textarea-autosize.js';
import { isFamilyFiltered } from '../auth.js';
import { chat, sessionStore as chatSessionStore } from '../chat.js';
import { getImageSettings } from '../image.js';
import { getModels, getVoices, getImageModels, getToolSettings, onChange as onCacheChange } from '../model-cache.js';
import { formatVoiceLabel } from '../voice-display.js';
import { openCardsmithLauncher } from './cardsmith.js';
import { createImageProgressLoader } from '../chat/image-progress.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let characters = [];       // Array of character objects (client-side / imported)
let activeCharId = null;   // Currently selected character
let narrativeState = null; // Latest server-side narrative state
let _lastStateFingerprint = ''; // Used to skip redundant re-renders
let pollTimer = null;
let currentBackgroundUrl = null; // Current auto-generated background image

// Persona state
let personas = [];
let editingPersonaId = null; // null = creating new, string = editing existing

// ---------------------------------------------------------------------------
// Character CRUD (server-backed with localStorage fallback)
// ---------------------------------------------------------------------------

const STORAGE_CHARS = 'augmentum_characters';
const STORAGE_CHARS_MIGRATED = 'augmentum_characters_migrated';
const AVATAR_MAX_SIZE = 256; // max width/height in pixels

/** Auto-resize a textarea to fit its content (min 2 rows, max 50vh).
 *  Delegates to the shared rAF-deferred helper so every keystroke
 *  doesn't force synchronous layout. See textarea-autosize.js for
 *  rationale.
 */
function autoResize(textarea) {
  scheduleAutosize(textarea);
}

/** Wire auto-resize to all textareas in a container. */
function wireAutoResize(container) {
  container.querySelectorAll('.field-textarea').forEach(ta => {
    autoResize(ta);
    ta.addEventListener('input', () => autoResize(ta));
  });
}

// ---------------------------------------------------------------------------
// Full-screen text editor modal
// ---------------------------------------------------------------------------

let expandModalEl = null;

function openExpandEditor(textarea, label) {
  if (!expandModalEl) {
    expandModalEl = document.createElement('div');
    expandModalEl.className = 'modal-overlay hidden';
    expandModalEl.id = 'expand-editor-overlay';
    expandModalEl.innerHTML = `
      <div class="expand-editor">
        <div class="expand-editor-bar">
          <span class="expand-editor-title"></span>
          <button class="btn btn-primary btn-sm" id="expand-editor-done">Done</button>
        </div>
        <textarea class="expand-editor-textarea" id="expand-editor-ta"></textarea>
      </div>
    `;
    document.body.appendChild(expandModalEl);

    expandModalEl.querySelector('#expand-editor-done').addEventListener('click', closeExpandEditor);
    expandModalEl.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeExpandEditor();
    });
  }

  expandModalEl._sourceTextarea = textarea;
  expandModalEl.querySelector('.expand-editor-title').textContent = label || 'Edit';
  const ta = expandModalEl.querySelector('#expand-editor-ta');
  ta.value = textarea.value;
  expandModalEl.classList.remove('hidden');
  ta.focus();
  // Place cursor at end
  ta.selectionStart = ta.selectionEnd = ta.value.length;
}

function closeExpandEditor() {
  if (!expandModalEl) return;
  const source = expandModalEl._sourceTextarea;
  const ta = expandModalEl.querySelector('#expand-editor-ta');
  if (source) {
    source.value = ta.value;
    source.dispatchEvent(new Event('input', { bubbles: true }));
    autoResize(source);
  }
  expandModalEl.classList.add('hidden');
  expandModalEl._sourceTextarea = null;
}

/** Add expand buttons to all field-group textareas in a container. */
function wireExpandButtons(container) {
  container.querySelectorAll('.field-group').forEach(group => {
    const ta = group.querySelector('.field-textarea');
    const label = group.querySelector('.field-label');
    if (!ta || group.querySelector('.expand-btn')) return;
    // Don't nest interactive elements inside <summary> (accessibility)
    if (label && label.tagName === 'SUMMARY') return;

    const btn = document.createElement('button');
    btn.className = 'expand-btn';
    btn.type = 'button';
    btn.title = 'Expand editor';
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>';
    btn.addEventListener('click', () => openExpandEditor(ta, label?.textContent || ''));
    // Insert after the label
    if (label) {
      label.style.display = 'flex';
      label.style.alignItems = 'center';
      label.style.justifyContent = 'space-between';
      label.appendChild(btn);
    }
  });
}

/**
 * Wire AI enhance buttons onto textareas in a container.
 * @param {HTMLElement} container - parent element containing .field-group elements
 * @param {string} contextType - "character" or "persona"
 * @param {Function} getContext - returns { name, fields: { field: value, ... } }
 */
function wireEnhanceButtons(container, contextType, getContext) {
  const fieldMap = {
    'char-desc': 'description',
    'char-personality': 'personality',
    'char-scenario': 'scenario',
    'char-greeting': 'greeting',
    'persona-appearance-input': 'appearance',
    'persona-description-input': 'description',
  };

  container.querySelectorAll('.field-group').forEach(group => {
    const ta = group.querySelector('.field-textarea');
    const label = group.querySelector('.field-label');
    if (!ta || !ta.id || !fieldMap[ta.id] || group.querySelector('.enhance-btn')) return;

    const fieldName = fieldMap[ta.id];

    const btn = document.createElement('button');
    btn.className = 'enhance-btn';
    btn.type = 'button';
    btn.title = 'Enhance with AI';
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>';

    btn.addEventListener('click', async () => {
      const text = ta.value.trim();
      if (!text) {
        showToast('Type something first — even brief notes work', 'warning');
        return;
      }

      const ctx = getContext();
      btn.classList.add('enhancing');
      btn.disabled = true;

      try {
        const resp = await fetch('/api/ui/enhance-field', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            field: fieldName,
            content: text,
            context_name: ctx.name || '',
            context_fields: ctx.fields || {},
          }),
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.error || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        if (data.enhanced) {
          ta.value = data.enhanced;
          ta.dispatchEvent(new Event('input', { bubbles: true }));
          autoResize(ta);
          showToast(`${fieldName.charAt(0).toUpperCase() + fieldName.slice(1)} enhanced`, 'success');
        }
      } catch (err) {
        showToast('Enhancement failed: ' + err.message, 'error');
      } finally {
        btn.classList.remove('enhancing');
        btn.disabled = false;
      }
    });

    if (label) {
      label.appendChild(btn);
    }
  });
}

/** Resize an image (from data URL or blob URL) to a small JPEG data URL. */
/**
 * Preview-then-accept avatar generation modal.
 * - Inherits width/height/steps/cfg/sampler/seed/negative/preset/model from the
 *   image panel via getImageSettings() — same source every other image path uses.
 * - Builds an image-model-aware prompt via /api/ui/character-portrait-prompt,
 *   falling back to a template concat if the endpoint is unavailable.
 * - User can edit the prompt, regenerate until they like the result, then
 *   Accept — which runs the generated image through resizeAvatar() and hands
 *   it to onAccept (the renderCardEditor-local handler that owns char.avatar
 *   + saveCharacters + re-render).
 *
 * @param {object} char - Character object being edited
 * @param {(dataUrl: string) => void} onAccept - Called with the resized avatar
 */
function _openAvatarGenModal(char, onAccept) {
  const modal = document.createElement('div');
  modal.className = 'avatar-gen-modal';
  modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;z-index:10000';
  modal.innerHTML = `
    <div style="background:var(--bg-elevated);border:1px solid var(--border);border-radius:12px;padding:var(--space-lg);max-width:460px;width:90%;display:flex;flex-direction:column;gap:var(--space-md)">
      <div style="font-weight:600;font-size:var(--text-lg)">Generate Avatar — ${escapeHtml(char.name || 'Character')}</div>
      <label class="field-label" style="margin:0">Prompt <span style="font-weight:400;color:var(--text-muted);font-size:var(--text-xs)">(editable)</span></label>
      <textarea id="_avgen-prompt" class="field-textarea" rows="4" style="font-size:var(--text-sm)" placeholder="Building prompt…"></textarea>
      <div id="_avgen-preview" style="width:100%;aspect-ratio:1;background:var(--bg-sunken);border-radius:8px;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:var(--text-sm);overflow:hidden">Ready to generate</div>
      <div style="display:flex;gap:var(--space-sm);justify-content:flex-end">
        <button class="btn btn-ghost" id="_avgen-cancel">Cancel</button>
        <button class="btn" id="_avgen-regen">Generate</button>
        <button class="btn btn-primary" id="_avgen-accept" disabled>Use Avatar</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);

  const promptTa = modal.querySelector('#_avgen-prompt');
  const previewEl = modal.querySelector('#_avgen-preview');
  const cancelBtn = modal.querySelector('#_avgen-cancel');
  const regenBtn = modal.querySelector('#_avgen-regen');
  const acceptBtn = modal.querySelector('#_avgen-accept');
  let lastImageUrl = '';

  const close = () => modal.remove();
  cancelBtn.addEventListener('click', close);
  modal.addEventListener('click', (e) => { if (e.target === modal) close(); });

  // Build initial prompt via the LLM-assisted endpoint. On failure, fall
  // back to the old template so the button still works offline / with the
  // utility model unreachable.
  (async () => {
    try {
      const resp = await fetch('/api/ui/character-portrait-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: char.name || '',
          description: char.visualTraits || char.description || '',
          personality: char.personality || '',
          scenario: char.scenario || '',
          style: 'portrait',
        }),
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data.prompt) promptTa.value = data.prompt;
      }
    } catch { /* fall through to template */ }
    if (!promptTa.value) {
      const traits = char.visualTraits?.trim() || char.description?.trim() || '';
      const styleHint = char.imageStyle ? `, ${char.imageStyle} style` : '';
      promptTa.value = `Portrait of ${char.name}, ${traits}${styleHint}, character portrait, full face visible, plain background`;
    }
  })();

  async function runGenerate() {
    if (!promptTa.value.trim()) {
      showToast('Prompt is empty', 'warning');
      return;
    }
    regenBtn.disabled = true;
    acceptBtn.disabled = true;
    regenBtn.textContent = 'Generating…';
    previewEl.innerHTML = 'Generating…';
    try {
      const p = getImageSettings();
      const body = {
        prompt: promptTa.value,
        model: char.imageModel || p.model || '',
        width: p.width || 512,
        height: p.height || 512,
      };
      if (p.steps)    body.steps = p.steps;
      if (p.cfg)      body.cfg_scale = p.cfg;
      if (p.sampler)  body.sampler = p.sampler;
      if (p.negative) body.negative_prompt = p.negative;
      if (p.preset)   body.preset = p.preset;
      if (p.seed > 0) body.seed = p.seed;

      const resp = await fetch('/api/image/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      lastImageUrl = data.url || data.image_url || '';
      if (!lastImageUrl) throw new Error('No image URL');
      previewEl.innerHTML = `<img src="${escapeHtml(lastImageUrl)}" style="width:100%;height:100%;object-fit:cover">`;
      acceptBtn.disabled = false;
      regenBtn.textContent = 'Regenerate';
    } catch (err) {
      previewEl.textContent = 'Generation failed';
      showToast('Avatar generation failed: ' + err.message, 'error');
      regenBtn.textContent = 'Generate';
    } finally {
      regenBtn.disabled = false;
    }
  }

  regenBtn.addEventListener('click', runGenerate);

  acceptBtn.addEventListener('click', async () => {
    if (!lastImageUrl) return;
    acceptBtn.disabled = true;
    acceptBtn.textContent = 'Saving…';
    try {
      const resized = await resizeAvatar(lastImageUrl);
      if (!resized) throw new Error('Failed to process image');
      onAccept(resized);
      close();
    } catch (err) {
      acceptBtn.disabled = false;
      acceptBtn.textContent = 'Use Avatar';
      showToast('Save failed: ' + err.message, 'error');
    }
  });
}

function resizeAvatar(srcUrl) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      let w = img.width, h = img.height;
      if (w > AVATAR_MAX_SIZE || h > AVATAR_MAX_SIZE) {
        const scale = AVATAR_MAX_SIZE / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }
      canvas.width = w;
      canvas.height = h;
      canvas.getContext('2d').drawImage(img, 0, 0, w, h);
      resolve(canvas.toDataURL('image/jpeg', 0.85));
    };
    img.onerror = () => resolve(null);
    img.src = srcUrl;
  });
}

/**
 * Enable drag-and-drop on an avatar area element.
 * @param {HTMLElement} area - The drop target element
 * @param {(dataUrl: string) => void} onDrop - Called with the resized data URL
 */
function enableAvatarDragDrop(area, onDrop) {
  area.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    area.classList.add('avatar-drag-over');
  });
  area.addEventListener('dragleave', () => {
    area.classList.remove('avatar-drag-over');
  });
  area.addEventListener('drop', async (e) => {
    e.preventDefault();
    area.classList.remove('avatar-drag-over');

    // 1. Dropped file (from file manager, desktop, etc.)
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      const resized = await resizeAvatar(url);
      URL.revokeObjectURL(url);
      if (resized) onDrop(resized);
      return;
    }

    // 2. Dragged image from browser / another app (URL or inline HTML)
    const imgUrl = e.dataTransfer.getData('text/uri-list')
                || e.dataTransfer.getData('text/plain')
                || '';
    if (imgUrl && /^https?:\/\/.+/i.test(imgUrl)) {
      const resized = await resizeAvatar(imgUrl);
      if (resized) onDrop(resized);
      return;
    }

    // 3. Dragged HTML with embedded <img> (e.g. from a web page)
    const html = e.dataTransfer.getData('text/html') || '';
    const match = html.match(/<img[^>]+src=["']([^"']+)["']/i);
    if (match?.[1]) {
      const resized = await resizeAvatar(match[1]);
      if (resized) onDrop(resized);
    }
  });
}

/**
 * Wire the per-character VRM picker (3D Avatar field-group in the editor).
 *
 * Shares the avatar library with Personalize — uploads from this surface
 * go through the same /api/avatar/upload endpoint, so a VRM added here
 * also appears in the Personalize cast. Pairing is bidirectional and
 * unfiltered: any avatar can be paired with any character, and a VRM
 * already paired to another character shows a "Paired with: X" subtitle
 * but is not blocked from re-pairing (last-write-wins by design).
 */
async function _wireCharVrmPicker(char, container) {
  const currentEl = container.querySelector('#char-vrm-current');
  const pickerEl = container.querySelector('#char-vrm-picker');
  const gridEl = container.querySelector('#char-vrm-picker-grid');
  const uploadInput = container.querySelector('#char-vrm-upload-input');
  if (!currentEl || !pickerEl || !gridEl || !uploadInput) return;

  let avatars = [];

  const reloadAvatars = async () => {
    try {
      const resp = await fetch('/api/avatar/list');
      if (!resp.ok) throw new Error('list failed');
      const data = await resp.json();
      avatars = (data.avatars || []).filter(a => a.type === 'vrm');
    } catch {
      avatars = [];
    }
  };

  const findPaired = () => avatars.find(a => a.character_id === char.id) || null;

  const charNameById = (cid) => {
    if (!cid) return null;
    if (cid === char.id) return char.name || 'this character';
    const c = characters.find(x => x.id === cid);
    return c?.name || null;
  };

  const renderCurrent = () => {
    const paired = findPaired();
    if (paired) {
      currentEl.innerHTML = `
        <div class="char-vrm-current-row">
          <img class="char-vrm-current-thumb" src="${escapeHtml(paired.thumbnail_url || '')}" alt="${escapeHtml(paired.name || 'Avatar')}" onerror="this.style.visibility='hidden'">
          <div class="char-vrm-current-meta">
            <div class="char-vrm-current-name">${escapeHtml(paired.name || 'Custom VRM')}</div>
            <div class="char-vrm-current-sub">${paired.is_bundled ? 'Bundled' : 'Your upload'}</div>
          </div>
          <button type="button" class="btn btn-sm" data-vrm-action="browse">${pickerEl.hidden ? 'Change' : 'Hide'}</button>
          <button type="button" class="btn btn-sm" data-vrm-action="unpair" style="color:var(--color-danger,#dc3545)">Unpair</button>
        </div>
      `;
    } else {
      currentEl.innerHTML = `
        <div class="char-vrm-current-row char-vrm-current-row--empty">
          <div class="char-vrm-current-meta">
            <div class="char-vrm-current-name" style="color:var(--text-muted)">No VRM paired</div>
            <div class="char-vrm-current-sub">Voice calls fall back to the user default avatar.</div>
          </div>
          <button type="button" class="btn btn-sm" data-vrm-action="browse">${pickerEl.hidden ? 'Browse VRMs' : 'Hide'}</button>
        </div>
      `;
    }
  };

  const renderGrid = () => {
    const paired = findPaired();
    // Bundled lead, then user uploads. Stable order across renders.
    const sorted = [...avatars].sort((a, b) => {
      if (!!a.is_bundled !== !!b.is_bundled) return a.is_bundled ? -1 : 1;
      return (a.created_at || '').localeCompare(b.created_at || '');
    });
    const cards = [];
    for (const av of sorted) {
      const isActive = paired?.id === av.id;
      const taken = av.character_id && av.character_id !== char.id;
      const takenName = taken ? charNameById(av.character_id) : null;
      const subtitle = isActive
        ? 'Currently paired'
        : takenName
          ? `Paired with ${escapeHtml(takenName)}`
          : (av.is_bundled ? 'Bundled' : 'Your upload');
      cards.push(`
        <button type="button" class="char-vrm-card${isActive ? ' is-active' : ''}" data-vrm-card="${escapeHtml(av.id)}">
          <div class="char-vrm-card-thumb">
            <img src="${escapeHtml(av.thumbnail_url || '')}" alt="${escapeHtml(av.name || 'Avatar')}" onerror="this.style.display='none'">
            ${av.is_bundled ? '<span class="char-vrm-card-badge-bundled">Bundled</span>' : ''}
            ${isActive ? '<span class="char-vrm-card-badge-active">✓</span>' : ''}
          </div>
          <div class="char-vrm-card-name">${escapeHtml(av.name || 'Custom VRM')}</div>
          <div class="char-vrm-card-sub">${subtitle}</div>
        </button>
      `);
    }
    cards.push(`
      <button type="button" class="char-vrm-card char-vrm-card-add" data-vrm-action="upload" title="Upload a .vrm file">
        <div class="char-vrm-card-thumb">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <line x1="12" y1="5" x2="12" y2="19"/>
            <line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
        </div>
        <div class="char-vrm-card-name">Upload new</div>
        <div class="char-vrm-card-sub">.vrm file</div>
      </button>
    `);
    gridEl.innerHTML = cards.join('');
  };

  const togglePicker = () => {
    pickerEl.hidden = !pickerEl.hidden;
    if (!pickerEl.hidden) renderGrid();
    renderCurrent();  // updates the "Change/Hide" button label
  };

  const pair = async (avatarId) => {
    try {
      const resp = await fetch(`/api/avatar/${encodeURIComponent(avatarId)}/character`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character_id: char.id }),
      });
      if (!resp.ok) throw new Error('pair failed');
      // Local-state patch — avoids a full /list refetch on every click.
      avatars = avatars.map(a => a.id === avatarId ? { ...a, character_id: char.id } : a);
      renderCurrent();
      renderGrid();
      showToast('Avatar paired', 'success');
    } catch {
      showToast('Failed to pair avatar', 'error');
    }
  };

  const unpair = async () => {
    const paired = findPaired();
    if (!paired) return;
    try {
      const resp = await fetch(`/api/avatar/${encodeURIComponent(paired.id)}/character`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character_id: null }),
      });
      if (!resp.ok) throw new Error('unpair failed');
      avatars = avatars.map(a => a.id === paired.id ? { ...a, character_id: null } : a);
      renderCurrent();
      if (!pickerEl.hidden) renderGrid();
      showToast('Avatar unpaired', 'success');
    } catch {
      showToast('Failed to unpair avatar', 'error');
    }
  };

  const upload = async (file) => {
    try {
      const form = new FormData();
      form.append('file', file);
      const resp = await fetch('/api/avatar/upload', { method: 'POST', body: form });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || 'Upload failed');
      }
      const result = await resp.json();
      const newId = result.avatar_id;
      if (!newId) throw new Error('Upload returned no avatar id');
      // Auto-pair the new VRM with this character so the user's intent
      // ("upload + use this for this character") completes in one gesture.
      await fetch(`/api/avatar/${encodeURIComponent(newId)}/character`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character_id: char.id }),
      });
      await reloadAvatars();
      renderCurrent();
      renderGrid();
      showToast('VRM uploaded and paired', 'success');
    } catch (e) {
      showToast(e.message || 'Upload failed', 'error');
    }
  };

  // Click delegation on the two render targets — render functions wipe
  // innerHTML on every state change, so binding to children would lose
  // listeners; binding to the container roots survives re-renders.
  currentEl.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-vrm-action]');
    if (!btn) return;
    const action = btn.dataset.vrmAction;
    if (action === 'browse') togglePicker();
    else if (action === 'unpair') unpair();
  });
  gridEl.addEventListener('click', (e) => {
    const card = e.target.closest('.char-vrm-card');
    if (!card) return;
    if (card.dataset.vrmAction === 'upload') {
      uploadInput.click();
      return;
    }
    if (card.dataset.vrmCard) {
      pair(card.dataset.vrmCard);
    }
  });
  uploadInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) upload(file);
    uploadInput.value = '';
  });

  await reloadAvatars();
  renderCurrent();
}

async function loadCharacters() {
  try {
    const resp = await fetch('/api/characters/');
    if (resp.ok) {
      const data = await resp.json();
      characters = data.characters || [];

      // One-time migration: push localStorage characters to server if not yet migrated
      if (!localStorage.getItem(STORAGE_CHARS_MIGRATED)) {
        const localRaw = localStorage.getItem(STORAGE_CHARS);
        if (localRaw) {
          const local = JSON.parse(localRaw);
          if (local.length > 0) {
            const serverIds = new Set(characters.map(c => c.id));
            const toMigrate = local.filter(c => !serverIds.has(c.id));
            if (toMigrate.length > 0) {
              await fetch('/api/characters/import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ characters: toMigrate }),
              });
              // Re-fetch merged list
              const resp2 = await fetch('/api/characters/');
              if (resp2.ok) {
                const data2 = await resp2.json();
                characters = data2.characters || [];
              }
            }
          }
        }
        localStorage.setItem(STORAGE_CHARS_MIGRATED, '1');
      }
      _rebaseCharacters();
      return;
    }
  } catch { /* server unavailable — fall back to localStorage */ }

  // Fallback: load from localStorage
  try {
    const raw = localStorage.getItem(STORAGE_CHARS);
    characters = raw ? JSON.parse(raw) : [];
  } catch { characters = []; }
}

let _charSaveTimer = null;
let _charDirtyIds = new Set(); // track which characters changed

function saveCharacters(changedId) {
  // Immediate localStorage backup for responsiveness.
  // Strip avatars and background images to avoid exceeding the ~5MB localStorage limit.
  // These are persisted on the server via the character sync endpoint.
  try {
    const lite = characters.map(c => {
      const copy = { ...c };
      delete copy.avatar;
      delete copy.backgroundImage;
      return copy;
    });
    localStorage.setItem(STORAGE_CHARS, JSON.stringify(lite));
  } catch {
    // If still too large, clear the localStorage backup entirely —
    // the server has the authoritative copy.
    localStorage.removeItem(STORAGE_CHARS);
  }

  // Track which character(s) need server sync
  if (changedId) {
    _charDirtyIds.add(changedId);
  } else {
    // Unknown — mark all as dirty
    for (const c of characters) _charDirtyIds.add(c.id);
  }

  // Stamp the edit so the server's stale-write guard can tell this change
  // apart from a concurrent one made on another device. `clientUpdatedAt`
  // is deliberately NOT `updatedAt` — the characters GET overwrites that
  // key with the server's ISO column for display.
  for (const id of _charDirtyIds) {
    const c = characters.find(x => x.id === id);
    if (c) c.clientUpdatedAt = Date.now();
  }

  // Debounced server sync (300ms)
  clearTimeout(_charSaveTimer);
  _charSaveTimer = setTimeout(_syncDirtyCharacters, 300);
}

// Per-card stamp of the copy we last saw on the server. Sent back as
// `baseUpdatedAt` so the server can answer "has anyone written since this
// client loaded?" — which catches a genuine concurrent edit, not just a
// stale tab replaying old content.
const _charBase = new Map();

function _rebaseCharacters() {
  _charBase.clear();
  for (const c of characters) {
    if (c && c.id) _charBase.set(c.id, Number(c.clientUpdatedAt) || 0);
  }
}

// A card was edited on two devices. There is no safe automatic answer —
// cards are a single blob, so unlike chat trees there is nothing to
// union-merge — so the user picks. The local edit is kept in memory
// either way, so "Keep theirs" is not destructive until they say so.
async function _onCharacterConflict(card) {
  const name = card?.name || 'this character';
  showChoiceToast(
    `"${name}" was changed on another device`,
    [
      {
        label: 'Keep mine',
        primary: true,
        onClick: async () => {
          // Re-read the server stamp, adopt it as our base, and re-push.
          // This is an explicit user-authorised overwrite, not a silent one.
          try {
            const resp = await fetch('/api/characters/');
            if (resp.ok) {
              const data = await resp.json();
              const theirs = (data.characters || []).find(c => c.id === card.id);
              _charBase.set(card.id, Number(theirs?.clientUpdatedAt) || 0);
            }
          } catch { /* offline — the retry below will 409 again, harmlessly */ }
          card.clientUpdatedAt = Date.now();
          _charDirtyIds.add(card.id);
          await _syncDirtyCharacters();
        },
      },
      {
        label: 'Keep theirs',
        onClick: async () => {
          await loadCharacters();
          await reloadAvatars();
          renderCurrent();
        },
      },
    ],
    {
      type: 'warning',
      description:
        'Your unsaved edit is still open here. Choose which version to keep.',
      dismissible: true,
    },
  );
}

async function _syncDirtyCharacters() {
  const ids = [..._charDirtyIds];
  _charDirtyIds.clear();
  if (ids.length === 0) return;

  try {
    // Sync only changed characters (avoids sending all avatars every time)
    const dirty = characters.filter(c => ids.includes(c.id));
    if (dirty.length === 0) return;

    if (dirty.length === 1) {
      // Single character — use PUT endpoint
      const card = dirty[0];
      const resp = await fetch(`/api/characters/${encodeURIComponent(card.id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...card,
          baseUpdatedAt: _charBase.get(card.id) || 0,
        }),
      });
      if (resp.status === 409) {
        // Someone else saved this card since we loaded it. Never silently
        // pick a winner — surface the choice (see the "never auto-select"
        // rule); the local edit stays in memory until the user decides.
        await _onCharacterConflict(card);
        return;
      }
      if (resp.ok) _charBase.set(card.id, Number(card.clientUpdatedAt) || 0);
    } else {
      // Multiple — use bulk import
      await fetch('/api/characters/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ characters: dirty }),
      });
      for (const c of dirty) {
        _charBase.set(c.id, Number(c.clientUpdatedAt) || 0);
      }
    }
  } catch { /* server unavailable — localStorage has the backup */ }
}

// Open the Cardsmith new-character flow. Routes the user through the
// launcher (Describe / Wiki / Blank). The Blank lane falls back to the
// legacy createCharacter() behavior. AI-Describe finalize flows through
// loadCharacters() + selectCharacter() so the freshly-saved server-side
// card lands in local state without losing fields.
function openNewCharacterFlow() {
  openCardsmithLauncher({
    onBlankRequested: () => createCharacter(),
    onCardSaved: async (charId, _name) => {
      try {
        await loadCharacters();
      } catch (err) {
        console.warn('[cardsmith] loadCharacters after save failed', err);
      }
      try { renderCharGrid(); } catch (err) { console.warn('[cardsmith] renderCharGrid failed', err); }
      if (charId) {
        // Switch the inspector to the card editor tab so the user sees
        // the freshly-saved card's editor — without this, renderCardEditor
        // runs but renders into a hidden #card-tab.
        try {
          const tabSelect = document.getElementById('inspector-section-select');
          if (tabSelect && tabSelect.value !== 'card-tab') {
            tabSelect.value = 'card-tab';
            tabSelect.dispatchEvent(new Event('change', { bubbles: true }));
          }
        } catch (err) { console.warn('[cardsmith] tab switch failed', err); }
        try {
          selectCharacter(charId, { openInspector: true });
        } catch (err) {
          console.error('[cardsmith] selectCharacter failed', err);
        }
        // Defensive: confirm the saved card is actually in the local state.
        // If it isn't, the GET /api/characters/ filter dropped it (most
        // likely user_id mismatch — e.g. session token went stale).
        const found = (typeof getCharacter === 'function') ? getCharacter(charId) : null;
        if (!found) {
          console.warn('[cardsmith] saved char_id not present in local characters[] after loadCharacters', { charId });
          showToast(
            'Card saved but not visible — refresh the page to see it.',
            'warning',
          );
        }
      }
    },
  });
}

function createCharacter(name = 'New Character') {
  const id = 'ch_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
  const char = {
    id,
    name,
    description: '',
    personality: '',
    scenario: '',
    greeting: '',
    alternateGreetings: [],
    examples: '',
    systemPrompt: '',
    postHistoryInstructions: '',
    depthPrompt: '',
    depthPromptDepth: 4,
    creatorNotes: '',
    tags: [],
    avatar: null,
    backgroundImage: null,
    visualTraits: '',   // AI-parsed or user-edited physical descriptors (images only)
    imageStyle: '',     // Art style for background generation (dropdown)
    voice: '',          // TTS voice ID for this character
    autoCollapseNarrativePanels: true,  // Default-collapse ```md/```stats/```scene panels in chat. False = expanded by default.
    lorebook: [],
    createdAt: Date.now(),
  };
  characters.push(char);
  saveCharacters();
  renderCharGrid();
  selectCharacter(id, { openInspector: true });
  return char;
}

function deleteCharacter(id) {
  characters = characters.filter(c => c.id !== id);
  if (activeCharId === id) {
    activeCharId = null;
    renderCardEditor(null);
    _reportScene(null);
  }
  saveCharacters();
  // Also delete from server directly
  fetch(`/api/characters/${encodeURIComponent(id)}`, { method: 'DELETE' }).catch(() => {});
  renderCharGrid();
  // Sessions referencing this character are intentionally preserved (see
  // _backfillSessionCharacterIds for the reattach-by-name path). Refresh the
  // strip so those chips re-render with the fallback placeholder avatar
  // instead of the now-stale character portrait.
  renderRecentChats();
}

function getCharacter(id) {
  return characters.find(c => c.id === id) || null;
}

// ---------------------------------------------------------------------------
// Character List (left panel — vertical list replacing grid)
// ---------------------------------------------------------------------------

// A session belongs to a character if its stored characterId matches.
// Legacy sessions (pre-id field) fall back to exact name match — but only
// when no other character also carries that name, to avoid cross-card leakage
// when duplicates exist.
function _sessionBelongsToChar(session, char) {
  if (!session || session.mode !== 'narrative' || !char) return false;
  if (session.characterId) return session.characterId === char.id;
  if (session.title !== char.name) return false;
  const dupes = characters.filter(c => c.name === char.name);
  return dupes.length === 1;
}

function getCharChatCount(char) {
  const sessions = chat.getSessions();
  if (!sessions) return 0;
  return Object.values(sessions).filter(s => _sessionBelongsToChar(s, char)).length;
}

function getCharSessions(char) {
  const sessions = chat.getSessions();
  if (!sessions) return [];
  return Object.values(sessions)
    .filter(s => _sessionBelongsToChar(s, char))
    .sort((a, b) => b.createdAt - a.createdAt);
}

// Resolve the character that owns a session. Prefers stored characterId;
// falls back to name match only when unambiguous.
function _charForSession(session) {
  if (!session) return null;
  if (session.characterId) {
    const byId = characters.find(c => c.id === session.characterId);
    if (byId) return byId;
  }
  if (!session.title) return null;
  const byName = characters.filter(c => c.name === session.title);
  return byName.length === 1 ? byName[0] : null;
}

// One-time backfill: for legacy narrative sessions with no characterId,
// set it from a unique name match. Ambiguous names are left alone so the
// user can reassign manually.
function _backfillSessionCharacterIds() {
  const sessions = chat.getSessions();
  if (!sessions) return;
  let mutated = false;
  for (const [id, s] of Object.entries(sessions)) {
    if (!s || s.mode !== 'narrative' || s.characterId || !s.title) continue;
    const matches = characters.filter(c => c.name === s.title);
    if (matches.length === 1) {
      s.characterId = matches[0].id;
      mutated = true;
      // Per-session save so each mutated session is marked dirty for
      // server sync (the debounce coalesces these into one request).
      chat.saveSessions?.(id);
    }
  }
  if (mutated) chat.saveSessions?.();
}

// Legacy group sessions (created before groupId was persisted on the session)
// only have a title matching the group.name. Backfill groupId by name where
// unambiguous so old chats keep working; requires `groups` to be loaded.
function _backfillSessionGroupIds() {
  const sessions = chat.getSessions();
  if (!sessions || !groups || groups.length === 0) return;
  let mutated = false;
  for (const [id, s] of Object.entries(sessions)) {
    if (!s || s.mode !== 'narrative' || s.groupId || !s.title) continue;
    const matches = groups.filter(g => g.name === s.title);
    if (matches.length === 1) {
      s.groupId = matches[0].id;
      s.groupMembers = matches[0].member_names;
      s.groupMode = matches[0].generation_mode;
      mutated = true;
      chat.saveSessions?.(id);
    }
  }
  if (mutated) chat.saveSessions?.();
}

// ---------------------------------------------------------------------------
// Recent Chats Strip
// ---------------------------------------------------------------------------

function renderRecentChats() {
  const strip = document.getElementById('recent-chats-strip');
  if (!strip) return;

  const sessions = chat.getSessions();
  if (!sessions) { strip.innerHTML = ''; return; }

  const activeId = chat.getActiveSessionId();

  // Get the most recent narrative sessions (up to 6)
  const recent = Object.values(sessions)
    .filter(s => s.mode === 'narrative' && s.title)
    .sort((a, b) => b.createdAt - a.createdAt)
    .slice(0, 6);

  if (recent.length === 0) { strip.innerHTML = ''; return; }

  strip.innerHTML = recent.map(s => {
    const char = _charForSession(s);
    const group = s.groupId ? groups.find(g => g.id === s.groupId) : null;
    const avatar = group?.avatar || char?.avatar;
    const isActive = s.id === activeId;
    // Orphan: session references a deleted character/group. A session with
    // no characterId AND no groupId is a plain narrative chat, not an orphan.
    const isOrphan = (s.characterId && !char) || (s.groupId && !group);
    const tooltip = isOrphan ? `${s.title} — original character/group was deleted` : s.title;
    const avatarHtml = avatar
      ? `<img class="recent-chat-chip-avatar" src="${escapeHtml(avatar)}" alt="">`
      : `<div class="recent-chat-chip-avatar-placeholder">${escapeHtml((s.title || '?').charAt(0).toUpperCase())}</div>`;
    const classes = ['recent-chat-chip'];
    if (isActive) classes.push('active');
    if (isOrphan) classes.push('orphan');
    return `<div class="${classes.join(' ')}" data-session-id="${escapeHtml(s.id)}" title="${escapeHtml(tooltip)}" role="button" tabindex="0">
      ${avatarHtml}
      <span class="recent-chat-chip-name">${escapeHtml(s.title)}</span>
      <button class="recent-chat-chip-delete" data-chip-delete="${escapeHtml(s.id)}" aria-label="Delete chat" title="Delete chat">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" width="10" height="10"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>`;
  }).join('');

  strip.querySelectorAll('.recent-chat-chip').forEach(chip => {
    const deleteBtn = chip.querySelector('.recent-chat-chip-delete');
    if (deleteBtn) {
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        // chat.deleteSession handles the confirm dialog, server delete, and
        // fires augmentum:session-deleted which re-renders the strip.
        chat.deleteSession(chip.dataset.sessionId);
      });
    }
    chip.addEventListener('click', (e) => {
      if (e.target.closest('.recent-chat-chip-delete')) return;
      chat.switchSession(chip.dataset.sessionId);
      // Sync character selection to match. If the session is orphaned (owner
      // character was deleted), clear the selection so the card panel doesn't
      // keep highlighting a character that no longer owns this chat.
      const session = chat.getActiveSession?.();
      if (session) {
        const char = _charForSession(session);
        if (char && char.id !== activeCharId) {
          selectCharacter(char.id);
        } else if (!char && activeCharId) {
          activeCharId = null;
          renderCardEditor(null);
          _reportScene(null);
        }
      }
      renderRecentChats();
      renderCharGrid();
    });
    chip.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        chip.click();
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Character Grid with Inline Chat Accordion
// ---------------------------------------------------------------------------

function renderCharGrid() {
  const grid = document.getElementById('char-grid');
  if (!grid) return;
  grid.innerHTML = '';

  // Update count badge
  const countEl = document.getElementById('char-count');
  if (countEl) countEl.textContent = characters.length;

  const activeSessionId = chat.getActiveSessionId();

  characters.forEach(c => {
    const isActive = c.id === activeCharId;
    const wrapper = document.createElement('div');
    wrapper.className = 'char-card-wrapper';
    wrapper.dataset.charId = c.id;

    const card = document.createElement('div');
    card.className = 'char-card' + (isActive ? ' active' : '');
    card.dataset.charId = c.id;

    const chatCount = getCharChatCount(c);
    const avatarContent = c.avatar
      ? `<img class="char-avatar" src="${escapeHtml(c.avatar)}" alt="${escapeHtml(c.name)}">`
      : `<div class="char-avatar" style="display:flex;align-items:center;justify-content:center;font-size:var(--text-sm);font-weight:600;color:var(--text-muted)">${escapeHtml(c.name.charAt(0).toUpperCase())}</div>`;

    card.innerHTML = `
      ${avatarContent}
      <span class="char-name">${escapeHtml(c.name)}</span>
      ${chatCount > 0 ? `<span class="char-chat-count">${chatCount}</span>` : ''}
      <div class="char-actions">
        <button class="char-action-btn char-edit-btn" title="Edit character">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
        </button>
        <button class="char-action-btn char-new-chat-btn" title="New chat">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><line x1="12" y1="8" x2="12" y2="14"/><line x1="9" y1="11" x2="15" y2="11"/></svg>
        </button>
      </div>
    `;

    wrapper.appendChild(card);

    // Inline chat accordion (only for active character)
    if (isActive) {
      const charSessions = getCharSessions(c);
      if (charSessions.length > 0 || true) {
        const accordion = document.createElement('div');
        accordion.className = 'char-inline-chats';

        let html = '';
        charSessions.forEach(s => {
          const date = new Date(s.createdAt);
          const dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
          const msgCount = s.tree ? Object.keys(s.tree).length : (s.messageCount || 0);
          const isCurrent = s.id === activeSessionId;
          html += `<div class="char-inline-chat${isCurrent ? ' active' : ''}" data-session-id="${escapeHtml(s.id)}">
            <svg class="char-inline-chat-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span class="char-inline-chat-title">${msgCount} msg${msgCount !== 1 ? 's' : ''}</span>
            <span class="char-inline-chat-date">${dateStr}</span>
            <div class="char-inline-chat-actions">
              <button class="message-action-btn" data-export-session="${escapeHtml(s.id)}" title="Export">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="10" height="10"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </button>
              <button class="message-action-btn" data-delete-session="${escapeHtml(s.id)}" title="Delete">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="10" height="10"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>
          </div>`;
        });

        // New chat + import buttons
        html += `<div style="display:flex;gap:2px">
          <button class="char-inline-new-chat" data-new-chat-for="${escapeHtml(c.id)}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            New chat
          </button>
          <button class="char-inline-new-chat" data-import-chat-for="${escapeHtml(c.id)}" title="Import chat">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Import
          </button>
        </div>`;

        accordion.innerHTML = html;

        // Wire chat item clicks
        accordion.querySelectorAll('.char-inline-chat').forEach(item => {
          item.addEventListener('click', (e) => {
            if (e.target.closest('[data-delete-session]') || e.target.closest('[data-export-session]')) return;
            chat.switchSession(item.dataset.sessionId);
            renderCharGrid();
            renderRecentChats();
          });
        });

        accordion.querySelectorAll('[data-export-session]').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            exportNarrativeChat(btn.dataset.exportSession);
          });
        });

        accordion.querySelectorAll('[data-delete-session]').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            chat.deleteSession(btn.dataset.deleteSession);
            renderCharGrid();
            renderRecentChats();
          });
        });

        accordion.querySelectorAll('[data-new-chat-for]').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            startChatWithCharacter(c);
            renderRecentChats();
          });
        });

        accordion.querySelectorAll('[data-import-chat-for]').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            importNarrativeChat();
          });
        });

        wrapper.appendChild(accordion);
      }
    }

    // Edit — open card editor in inspector
    card.querySelector('.char-edit-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      selectCharacter(c.id, { openInspector: true });
    });

    // New Chat — start a fresh session with this character
    card.querySelector('.char-new-chat-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      selectCharacter(c.id);
      startChatWithCharacter(c);
      renderRecentChats();
    });

    // Clicking the card body toggles selection (click again to deselect)
    card.addEventListener('click', () => {
      if (activeCharId === c.id) {
        activeCharId = null;
        renderCardEditor(null);
        renderCharGrid();
        renderRecentChats();
        _reportScene(null);
      } else {
        selectCharacter(c.id);
      }
    });
    grid.appendChild(wrapper);
  });
}

// Companion presence: which story scene is active. Fires on select
// (scene_active with the character) and deselect/delete (scene_closed)
// so "this scene" / "this character" deixis has a referent server-side.
// Best-effort + dedup-windowed in the observer; safe to call freely.
function _reportScene(char) {
  import('../architect-observer.js')
    .then(m => char
      ? m.reportAttention('surface.narrative.scene_active', {
          label: char.name || '', ref: char.id || '',
        })
      : m.reportAttention('surface.narrative.scene_closed', {}))
    .catch(() => {});
}

function selectCharacter(id, { openInspector = false } = {}) {
  activeCharId = id;
  renderCharGrid();
  renderRecentChats();
  const char = getCharacter(id);
  renderCardEditor(char);
  applyCharBackground(char);
  _reportScene(char);
  // Push narrative-panel preference to the renderer so future messages use
  // this character's default-collapsed setting. Safe to call with no char
  // (defaults to true).
  if (chat.setNarrativePanelsCollapsed) {
    chat.setNarrativePanelsCollapsed(char?.autoCollapseNarrativePanels !== false);
  }

  // Refresh whichever inspector tab is currently visible so it picks up
  // the new character's data (portrait, LTM, lore, etc.)
  const tabSelect = document.getElementById('inspector-section-select');
  const activeTab = tabSelect?.value;
  if (activeTab === 'portrait-tab') updatePortraitTab();
  else if (activeTab === 'lore-tab') renderLorebook();

  // When the user explicitly asked to view the editor (the Edit pencil
  // button passes openInspector:true):
  //   1. Make sure the inspector PANEL itself is visible (desktop: it can
  //      be collapsed via the toggle; the data-inspector attr on the app
  //      root tells us). Without this, the editor renders into a hidden
  //      #card-tab inside a hidden inspector and the user sees nothing.
  //   2. Switch the inspector tab to card-tab so the editor is the active
  //      tab content.
  if (openInspector) {
    const isDesktop = window.innerWidth >= 1024;
    if (isDesktop) {
      const appRoot = app?.dom?.app || document.documentElement;
      const visible = appRoot.getAttribute('data-inspector') === 'visible';
      if (!visible && typeof app?.toggleInspector === 'function') {
        try { app.toggleInspector(); } catch (err) { console.warn('[narrative] toggleInspector failed', err); }
      }
    }
    if (tabSelect && tabSelect.value !== 'card-tab') {
      tabSelect.value = 'card-tab';
      tabSelect.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }

  // On mobile/tablet, open inspector overlay — but only on explicit user action,
  // not during initial page load (openInspector flag)
  if (openInspector && window.innerWidth < 1024) {
    app.closeImagePanel();
    if (window.innerWidth < 768) {
      app.closePanel();
    }
    app.openInspectorMobile();
  }
}

// ---------------------------------------------------------------------------
// Character Background Image
// ---------------------------------------------------------------------------

function applyCharBackground(char) {
  const mainArea = document.querySelector('.main-area');
  if (!mainArea) return;

  let bgEl = mainArea.querySelector('.chat-bg-image');
  if (!bgEl) {
    bgEl = document.createElement('div');
    bgEl.className = 'chat-bg-image';
    mainArea.prepend(bgEl);
  }

  if (char?.backgroundImage) {
    bgEl.style.backgroundImage = `url(${char.backgroundImage})`;
    // Small delay for fade-in transition
    requestAnimationFrame(() => bgEl.classList.add('active'));
  } else {
    bgEl.classList.remove('active');
    // Clear after fade-out
    setTimeout(() => {
      if (!bgEl.classList.contains('active')) {
        bgEl.style.backgroundImage = '';
      }
    }, 800);
  }
}

// ---------------------------------------------------------------------------
// Character Chat History (left panel — below character list)
// ---------------------------------------------------------------------------

function renderCharChatList(char) {
  const section = document.getElementById('narrative-chats-section');
  const list = document.getElementById('char-chat-list');
  const label = document.getElementById('char-chats-label');
  if (!section || !list) return;

  if (!char) {
    section.style.display = 'none';
    return;
  }

  section.style.display = '';
  if (label) label.textContent = `${char.name} Chats`;

  const sessions = chat.getSessions();
  if (!sessions) {
    list.innerHTML = '<div class="char-chat-empty">No chats yet</div>';
    return;
  }

  const charSessions = Object.values(sessions)
    .filter(s => s.mode === 'narrative' && s.title === char.name)
    .sort((a, b) => b.createdAt - a.createdAt);

  if (charSessions.length === 0) {
    list.innerHTML = '<div class="char-chat-empty">No chats yet. Click + to start one.</div>';
    return;
  }

  const activeId = chat.getActiveSessionId();

  list.innerHTML = charSessions.map(s => {
    const date = new Date(s.createdAt);
    const dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    const isActive = s.id === activeId;

    // Count messages in tree
    const msgCount = s.tree ? Object.keys(s.tree).length : 0;

    return `
      <div class="char-chat-item${isActive ? ' active' : ''}" data-session-id="${escapeHtml(s.id)}">
        <svg class="char-chat-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <span class="char-chat-item-title">${msgCount} message${msgCount !== 1 ? 's' : ''}</span>
        <span class="char-chat-item-date">${dateStr}</span>
        <button class="message-action-btn char-chat-item-export" data-export-session="${escapeHtml(s.id)}" title="Export">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        </button>
        <button class="message-action-btn char-chat-item-delete" data-delete-session="${escapeHtml(s.id)}" title="Delete">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
    `;
  }).join('');

  // Wire click handlers
  list.querySelectorAll('.char-chat-item').forEach(item => {
    item.addEventListener('click', (e) => {
      if (e.target.closest('[data-delete-session]') || e.target.closest('[data-export-session]')) return;
      chat.switchSession(item.dataset.sessionId);
      renderCharChatList(char);
    });
  });

  list.querySelectorAll('[data-export-session]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      exportNarrativeChat(btn.dataset.exportSession);
    });
  });

  list.querySelectorAll('[data-delete-session]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      chat.deleteSession(btn.dataset.deleteSession);
      renderCharChatList(char);
      renderCharGrid();
    });
  });
}

// ---------------------------------------------------------------------------
// Character Card Editor (inspector panel — card-tab)
// ---------------------------------------------------------------------------

function renderCardEditorEmpty() {
  const container = document.getElementById('card-tab');
  if (!container) return;
  container.innerHTML = `
    <div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
      <p>No characters yet. Create or import one to get started.</p>
      <div style="display:flex;gap:var(--space-sm);margin-top:var(--space-sm)">
        <button class="btn btn-sm btn-primary" id="card-empty-create">Create</button>
        <button class="btn btn-sm" id="card-empty-import">Import</button>
      </div>
    </div>`;
  const createBtn = container.querySelector('#card-empty-create');
  const importBtn = container.querySelector('#card-empty-import');
  if (createBtn) createBtn.addEventListener('click', () => openNewCharacterFlow());
  if (importBtn) importBtn.addEventListener('click', openImportDialog);
}

function renderCardEditor(char) {
  const container = document.getElementById('card-tab');
  if (!container) return;

  if (!char) {
    container.innerHTML = `
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
          <circle cx="12" cy="7" r="4"/>
        </svg>
        <p>Select a character to view their card.</p>
      </div>`;
    return;
  }

  // Build alternate greetings section
  const altGreetings = char.alternateGreetings || [];
  let altGreetingsHtml = '';
  if (altGreetings.length > 0) {
    altGreetingsHtml = `
      <div class="field-group">
        <label class="field-label">Alternate Greetings (${altGreetings.length})</label>
        <div id="alt-greetings-list" style="display:flex;flex-direction:column;gap:var(--space-xs)">
          ${altGreetings.map((g, i) => `
            <div class="alt-greeting-item" style="display:flex;gap:var(--space-xs);align-items:flex-start">
              <textarea class="field-textarea alt-greeting-text" rows="2" data-idx="${i}" style="flex:1">${escapeHtml(g)}</textarea>
              <div style="display:flex;flex-direction:column;gap:2px">
                <button class="btn btn-sm use-greeting-btn" data-idx="${i}" title="Use as greeting" style="font-size:var(--text-xs);padding:2px 6px">Use</button>
                <button class="btn btn-sm del-greeting-btn" data-idx="${i}" title="Remove" style="font-size:var(--text-xs);padding:2px 6px;color:var(--error)">Del</button>
              </div>
            </div>
          `).join('')}
        </div>
        <button class="btn btn-sm" id="add-alt-greeting-btn" style="margin-top:var(--space-xs)">Add Alternate</button>
      </div>`;
  } else {
    altGreetingsHtml = `
      <div class="field-group">
        <label class="field-label">Alternate Greetings</label>
        <button class="btn btn-sm" id="add-alt-greeting-btn">Add Alternate Greeting</button>
      </div>`;
  }

  container.innerHTML = `
    <div class="char-editor">
      <div class="char-editor-header">
        <div class="char-editor-avatar" id="char-avatar-area" style="display:flex;align-items:center;justify-content:center;font-size:var(--text-2xl);font-weight:600;color:var(--text-muted);position:relative" title="Click to upload avatar">
          ${char.avatar ? `<img src="${escapeHtml(char.avatar)}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit">` : escapeHtml(char.name.charAt(0).toUpperCase())}
          <div style="position:absolute;bottom:-2px;right:-2px;width:22px;height:22px;background:var(--accent);border-radius:50%;display:flex;align-items:center;justify-content:center;border:2px solid var(--bg-elevated)">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" width="12" height="12"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          </div>
          <input type="file" id="char-avatar-input" accept="image/*" style="display:none">
        </div>
        <div class="char-editor-name">
          <input type="text" class="field-input" id="char-name-input" value="${escapeHtml(char.name)}" placeholder="Character name">
          <button class="btn btn-sm" id="char-gen-avatar-btn" title="Generate a portrait avatar from visual traits" style="margin-top:var(--space-xs);width:100%;font-size:var(--text-xs)">Generate Avatar</button>
        </div>
      </div>

      <div class="field-group">
        <label class="field-label">3D Avatar (VRM)</label>
        <div class="settings-desc" style="margin-bottom:var(--space-sm)">Pair this character with a VRM model — appears in voice calls and anywhere the AI shows up. Avatars are shared with Personalize; uploads from here also appear there.</div>
        <div id="char-vrm-current" class="char-vrm-current"></div>
        <div id="char-vrm-picker" class="char-vrm-picker" hidden>
          <div id="char-vrm-picker-grid" class="char-vrm-picker-grid"></div>
        </div>
        <input type="file" id="char-vrm-upload-input" accept=".vrm" hidden>
      </div>

      <div class="field-group">
        <label class="field-label" style="display:flex;align-items:center;gap:var(--space-xs)">
          Visual Traits <span style="font-weight:400;font-size:var(--text-xs);color:var(--text-muted)">(images only)</span>
          <button class="btn btn-sm" id="char-extract-traits-btn" title="AI-parse visual traits from description" style="margin-left:auto;font-size:var(--text-xs);padding:2px 8px">Extract</button>
        </label>
        <textarea class="field-textarea" id="char-visual-traits" rows="3" placeholder="Single: blonde hair, blue eyes, athletic build, school uniform&#10;Ensemble: <Alice> red hair, green eyes, freckles <Bob> tall, dark skin, glasses&#10;World/RPG: medieval fantasy, stone castles, enchanted forests, torchlit dungeons">${escapeHtml(char.visualTraits || '')}</textarea>
      </div>

      <div class="field-group">
        <label class="field-label">Image Style <span style="font-weight:400;font-size:var(--text-xs);color:var(--text-muted)">(auto backgrounds)</span></label>
        <select class="field-input" id="char-image-style">
          <option value="">Auto (no style hint)</option>
          <option value="anime"${char.imageStyle === 'anime' ? ' selected' : ''}>Anime / Manga</option>
          <option value="painterly"${char.imageStyle === 'painterly' ? ' selected' : ''}>Painterly / Concept Art</option>
          <option value="photorealistic"${char.imageStyle === 'photorealistic' ? ' selected' : ''}>Photorealistic / Cinematic</option>
          <option value="watercolor"${char.imageStyle === 'watercolor' ? ' selected' : ''}>Watercolor / Soft Illustration</option>
          <option value="pixel"${char.imageStyle === 'pixel' ? ' selected' : ''}>Pixel Art / Retro</option>
          <option value="comic"${char.imageStyle === 'comic' ? ' selected' : ''}>Comic Book / Graphic Novel</option>
          <option value="dark"${char.imageStyle === 'dark' ? ' selected' : ''}>Dark / Gothic / Horror</option>
          <option value="fantasy"${char.imageStyle === 'fantasy' ? ' selected' : ''}>High Fantasy / Epic</option>
          <option value="scifi"${char.imageStyle === 'scifi' ? ' selected' : ''}>Sci-Fi / Cyberpunk</option>
          <option value="ukiyoe"${char.imageStyle === 'ukiyoe' ? ' selected' : ''}>Ukiyo-e / East Asian Ink</option>
          <option value="noir"${char.imageStyle === 'noir' ? ' selected' : ''}>Film Noir / Monochrome</option>
          <option value="cozy"${char.imageStyle === 'cozy' ? ' selected' : ''}>Cozy / Slice of Life</option>
        </select>
      </div>

      <div class="field-group">
        <label class="field-label">Description</label>
        <textarea class="field-textarea" id="char-desc" rows="3" placeholder="Character appearance, traits, background...">${escapeHtml(char.description)}</textarea>
      </div>

      <div class="field-group">
        <label class="field-label">Personality</label>
        <textarea class="field-textarea" id="char-personality" rows="3" placeholder="Personality traits, speech patterns...">${escapeHtml(char.personality)}</textarea>
      </div>

      <div class="field-group">
        <label class="field-label">Scenario</label>
        <textarea class="field-textarea" id="char-scenario" rows="2" placeholder="Setting and situation...">${escapeHtml(char.scenario)}</textarea>
      </div>

      <div class="field-group">
        <label class="field-label">Greeting</label>
        <textarea class="field-textarea" id="char-greeting" rows="3" placeholder="First message from this character...">${escapeHtml(char.greeting)}</textarea>
      </div>

      ${altGreetingsHtml}

      <div class="field-group">
        <label class="field-label">Example Messages</label>
        <textarea class="field-textarea" id="char-examples" rows="3" placeholder="<START>&#10;{{user}}: ...&#10;{{char}}: ...">${escapeHtml(char.examples)}</textarea>
      </div>

      <div class="field-group">
        <label class="field-label">Creator Notes / Author's Note</label>
        <textarea class="field-textarea" id="char-creator-notes" rows="2" placeholder="Writing style instructions, story guidance...">${escapeHtml(char.creatorNotes || '')}</textarea>
      </div>

      <details class="field-group char-advanced-section">
        <summary class="field-label" style="cursor:pointer;user-select:none">Advanced Card Fields</summary>
        <div style="display:flex;flex-direction:column;gap:var(--space-sm);margin-top:var(--space-sm)">
          <div class="field-group" style="margin:0">
            <label class="field-label" style="font-size:var(--text-xs)">System Prompt</label>
            <textarea class="field-textarea" id="char-system-prompt" rows="2" placeholder="Custom system instructions for this character...">${escapeHtml(char.systemPrompt || '')}</textarea>
          </div>
          <div class="field-group" style="margin:0">
            <label class="field-label" style="font-size:var(--text-xs)">Post-History Instructions</label>
            <textarea class="field-textarea" id="char-post-history" rows="2" placeholder="Injected before the last user message...">${escapeHtml(char.postHistoryInstructions || '')}</textarea>
          </div>
          <div style="display:flex;gap:var(--space-sm);align-items:flex-end">
            <div class="field-group" style="margin:0;flex:1">
              <label class="field-label" style="font-size:var(--text-xs)">Depth Prompt</label>
              <textarea class="field-textarea" id="char-depth-prompt" rows="2" placeholder="Author's note injected at a specific depth...">${escapeHtml(char.depthPrompt || '')}</textarea>
            </div>
            <div class="field-group" style="margin:0;width:60px">
              <label class="field-label" style="font-size:var(--text-xs)">Depth</label>
              <input type="number" class="field-input" id="char-depth-prompt-depth" value="${char.depthPromptDepth ?? 4}" min="0" max="100" style="text-align:center">
            </div>
          </div>
          <div class="field-group" style="margin:0">
            <label class="field-label" style="font-size:var(--text-xs)">Tags</label>
            <input type="text" class="field-input" id="char-tags" value="${escapeHtml((char.tags || []).join(', '))}" placeholder="tag1, tag2, tag3...">
          </div>
          <div class="field-group" style="margin:0">
            <label style="display:flex;align-items:center;gap:var(--space-xs);cursor:pointer;font-size:var(--text-xs)">
              <input type="checkbox" id="char-auto-collapse-panels" ${char.autoCollapseNarrativePanels !== false ? 'checked' : ''}>
              <span>Auto-collapse narrative panels (<code>\`\`\`md</code>, <code>\`\`\`stats</code>, <code>\`\`\`scene</code>) by default</span>
            </label>
            <div style="font-size:var(--text-xs);color:var(--text-muted);margin-top:2px;line-height:1.4">
              Many cards emit stat/scene blocks every turn for the LLM's own bookkeeping.
              Collapsed by default keeps chat readable; uncheck to always show their contents.
            </div>
          </div>
        </div>
      </details>

      <div class="field-group">
        <label class="field-label">Background Image</label>
        <div style="display:flex;gap:var(--space-sm);align-items:center">
          <input type="text" class="field-input" id="char-bg-url" value="${escapeHtml(char.backgroundImage || '')}" placeholder="URL or drop an image..." style="flex:1">
          <button class="btn btn-sm" id="char-bg-upload-btn" title="Upload image">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          </button>
          <input type="file" id="char-bg-file" accept="image/*" style="display:none">
          ${char.backgroundImage ? '<button class="btn btn-sm" id="char-bg-clear-btn" title="Remove" style="color:var(--error)">Clear</button>' : ''}
        </div>
        ${char.backgroundImage ? `<div style="margin-top:var(--space-xs);height:60px;border-radius:var(--radius-sm);overflow:hidden;background:url(${escapeHtml(char.backgroundImage)}) center/cover no-repeat;border:1px solid var(--border)"></div>` : ''}
      </div>

      <div class="field-group">
        <label class="field-label">Voice (TTS)</label>
        <div style="display:flex;gap:var(--space-sm);align-items:center">
          <select class="field-input" id="char-voice-select" style="flex:1">
            <option value="">Default (provider default)</option>
          </select>
          <button class="btn btn-sm" id="char-voice-preview-btn" title="Preview voice">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
          </button>
        </div>
      </div>

      <div style="display:flex;gap:var(--space-sm);flex-wrap:wrap;position:relative;z-index:1">
        <button type="button" class="btn btn-primary btn-sm" id="char-save-btn">Save Changes</button>
        <div class="char-translate-wrap" style="position:relative;display:inline-block">
          <button type="button" class="btn btn-sm" id="char-translate-btn" title="Translate card fields with your LLM">Translate</button>
          <div class="char-translate-popover hidden" id="char-translate-popover" role="dialog" aria-label="Translate card">
            <div class="char-translate-popover-header">Translate card</div>
            <div class="char-translate-popover-row">
              <label for="char-translate-target">Translate to</label>
              <select id="char-translate-target" class="char-translate-select">
                <option value="English">English</option>
                <option value="Español">Español</option>
                <option value="Français">Français</option>
                <option value="Deutsch">Deutsch</option>
                <option value="Italiano">Italiano</option>
                <option value="Português">Português</option>
                <option value="Nederlands">Nederlands</option>
                <option value="Polski">Polski</option>
                <option value="Русский">Русский</option>
                <option value="Türkçe">Türkçe</option>
                <option value="العربية">العربية</option>
                <option value="हिन्दी">हिन्दी</option>
                <option value="中文">中文</option>
                <option value="日本語">日本語</option>
                <option value="한국어">한국어</option>
                <option value="__custom__">Custom…</option>
              </select>
              <input type="text" id="char-translate-custom" class="char-translate-input hidden" placeholder="Language name" autocomplete="off">
            </div>
            <div class="char-translate-popover-row">
              <label for="char-translate-source">Source language</label>
              <input type="text" id="char-translate-source" class="char-translate-input" placeholder="Auto-detect" autocomplete="off">
            </div>
            <div class="char-translate-popover-row">
              <label class="char-translate-check"><input type="checkbox" id="char-translate-preview" checked> Preview changes before applying</label>
            </div>
            <div class="char-translate-popover-actions">
              <button type="button" class="btn btn-sm" id="char-translate-cancel">Cancel</button>
              <button type="button" class="btn btn-primary btn-sm" id="char-translate-go">Translate</button>
            </div>
          </div>
        </div>
        <button type="button" class="btn btn-sm" id="char-export-json-btn">Export JSON</button>
        <button type="button" class="btn btn-sm" id="char-export-png-btn">Export PNG</button>
        <button type="button" class="btn btn-sm" id="char-start-chat-btn" style="color:var(--accent)">New Chat</button>
        <button type="button" class="btn btn-sm" id="char-delete-btn" style="color:var(--error)">Delete</button>
      </div>
    </div>
  `;

  // Auto-resize textareas to fit content
  wireAutoResize(container);
  wireExpandButtons(container);

  // AI enhance buttons for text fields
  wireEnhanceButtons(container, 'character', () => ({
    name: container.querySelector('#char-name-input')?.value || char.name,
    fields: {
      description: container.querySelector('#char-desc')?.value || '',
      personality: container.querySelector('#char-personality')?.value || '',
      scenario: container.querySelector('#char-scenario')?.value || '',
      greeting: container.querySelector('#char-greeting')?.value || '',
    },
  }));

  // Collect all field values from the editor into the char object
  function syncFieldsToChar() {
    char.name = container.querySelector('#char-name-input').value.trim() || 'Unnamed';
    char.visualTraits = container.querySelector('#char-visual-traits')?.value || '';
    char.imageStyle = container.querySelector('#char-image-style')?.value || '';
    char.description = container.querySelector('#char-desc').value;
    char.personality = container.querySelector('#char-personality').value;
    char.scenario = container.querySelector('#char-scenario').value;
    char.greeting = container.querySelector('#char-greeting').value;
    char.examples = container.querySelector('#char-examples').value;
    char.creatorNotes = container.querySelector('#char-creator-notes').value;
    char.systemPrompt = container.querySelector('#char-system-prompt')?.value || '';
    char.postHistoryInstructions = container.querySelector('#char-post-history')?.value || '';
    char.depthPrompt = container.querySelector('#char-depth-prompt')?.value || '';
    char.depthPromptDepth = parseInt(container.querySelector('#char-depth-prompt-depth')?.value) || 4;
    char.tags = (container.querySelector('#char-tags')?.value || '').split(',').map(t => t.trim()).filter(Boolean);
    char.backgroundImage = container.querySelector('#char-bg-url')?.value?.trim() || null;
    char.voice = container.querySelector('#char-voice-select')?.value || '';
    const collapseCheck = container.querySelector('#char-auto-collapse-panels');
    if (collapseCheck) char.autoCollapseNarrativePanels = collapseCheck.checked;
    // Sync voice to active chat session so TTS picks it up immediately
    const activeId = chatSessionStore.getActiveId();
    if (activeId) chat.updateCharacterVoice(activeId, char.voice);

    // Save alternate greetings — filter empties and duplicates of primary
    const altTexts = container.querySelectorAll('.alt-greeting-text');
    if (altTexts.length > 0) {
      const primary = char.greeting || '';
      char.alternateGreetings = Array.from(altTexts)
        .map(t => t.value)
        .filter(g => g && g.trim() && g !== primary);
    }
  }

  // Debounced auto-save on any field change
  let autoSaveTimer = null;
  function autoSave() {
    clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(() => {
      syncFieldsToChar();
      saveCharacters(char.id);
    }, 800);
  }
  container.querySelectorAll('textarea, input[type="text"], input[type="number"], select').forEach(el => {
    el.addEventListener('input', autoSave);
    el.addEventListener('change', autoSave);
  });

  // Explicit save button (immediate, with toast)
  container.querySelector('#char-save-btn').addEventListener('click', () => {
    clearTimeout(autoSaveTimer);
    syncFieldsToChar();
    saveCharacters(char.id);
    renderCharGrid();
    showToast('Character saved', 'success');
  });

  // Visual traits AI extraction
  const extractBtn = container.querySelector('#char-extract-traits-btn');
  if (extractBtn) {
    extractBtn.addEventListener('click', async () => {
      const desc = container.querySelector('#char-desc')?.value || '';
      if (!desc.trim()) {
        showToast('Write a description first', 'warning');
        return;
      }
      extractBtn.disabled = true;
      extractBtn.textContent = 'Extracting...';
      try {
        const charName = container.querySelector('#char-name-input')?.value || '';
        const scenario = container.querySelector('#char-scenario')?.value || '';
        const personality = container.querySelector('#char-personality')?.value || '';
        const resp = await fetch('/api/image/extract-visual-traits', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ description: desc, name: charName, scenario, personality }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        const traitsField = container.querySelector('#char-visual-traits');
        if (traitsField && data.visual_traits) {
          traitsField.value = data.visual_traits;
          syncFieldsToChar();
          saveCharacters(char.id);
          showToast('Visual traits extracted & saved', 'success');
        }
      } catch (err) {
        showToast('Failed to extract traits: ' + err.message, 'error');
      } finally {
        extractBtn.disabled = false;
        extractBtn.textContent = 'Extract';
      }
    });
  }

  // Avatar upload handler (click + drag-and-drop)
  const avatarArea = container.querySelector('#char-avatar-area');
  const avatarInput = container.querySelector('#char-avatar-input');
  avatarArea.addEventListener('click', () => avatarInput.click());
  const _applyCharAvatar = (resized) => {
    char.avatar = resized;
    saveCharacters(char.id);
    renderCharGrid();
    renderCardEditor(char);
    showToast('Avatar updated', 'success');
  };
  avatarInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const resized = await resizeAvatar(url);
    URL.revokeObjectURL(url);
    if (resized) _applyCharAvatar(resized);
  });
  enableAvatarDragDrop(avatarArea, _applyCharAvatar);

  // 3D VRM pairing — bidirectional with Personalize's avatar library.
  // Uploads here go through the same /api/avatar/upload endpoint used in
  // Personalize, so any VRM added from this surface shows up there too.
  _wireCharVrmPicker(char, container);

  // Generate Avatar button — opens a preview modal that uses the user's
  // image panel settings (via getImageSettings) and an LLM-built prompt.
  // Preview-then-accept avoids silently overwriting the avatar with a bad
  // result; Regenerate lets the user iterate without leaving the modal.
  const genAvatarBtn = container.querySelector('#char-gen-avatar-btn');
  if (genAvatarBtn) {
    genAvatarBtn.addEventListener('click', () => {
      if (!(char.visualTraits?.trim() || char.description?.trim())) {
        showToast('Add a description or visual traits first', 'warning');
        return;
      }
      _openAvatarGenModal(char, _applyCharAvatar);
    });
  }

  // Background image upload
  const bgUploadBtn = container.querySelector('#char-bg-upload-btn');
  const bgFileInput = container.querySelector('#char-bg-file');
  if (bgUploadBtn && bgFileInput) {
    bgUploadBtn.addEventListener('click', () => bgFileInput.click());
    bgFileInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        char.backgroundImage = ev.target.result;
        saveCharacters();
        renderCardEditor(char);
        applyCharBackground(char);
        showToast('Background set', 'success');
      };
      reader.readAsDataURL(file);
    });
  }

  // Background clear
  container.querySelector('#char-bg-clear-btn')?.addEventListener('click', () => {
    char.backgroundImage = null;
    saveCharacters();
    renderCardEditor(char);
    applyCharBackground(null);
    showToast('Background removed', 'success');
  });

  // Voice dropdown — load available voices from TTS provider
  const voiceSelect = container.querySelector('#char-voice-select');
  if (voiceSelect) {
    _loadVoiceOptions(voiceSelect, char.voice || '');
  }

  // Voice preview
  container.querySelector('#char-voice-preview-btn')?.addEventListener('click', async () => {
    const selectedVoice = voiceSelect?.value || '';
    const previewText = `Hello, I'm ${char.name}.`;
    try {
      const resp = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: previewText,
          voice: selectedVoice || undefined,
        }),
      });
      if (!resp.ok) throw new Error(`TTS error: ${resp.status}`);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      audio.onerror = () => URL.revokeObjectURL(url);
      await audio.play();
    } catch (err) {
      showToast('Voice preview failed — is a TTS provider configured?', 'warning');
    }
  });

  // Export handlers
  container.querySelector('#char-export-json-btn').addEventListener('click', () => {
    exportCharacter(char);
  });
  container.querySelector('#char-export-png-btn').addEventListener('click', () => {
    exportCharacterAsPng(char);
  });

  // Start chat handler — creates a new session with the greeting as first message
  container.querySelector('#char-start-chat-btn').addEventListener('click', () => {
    startChatWithCharacter(char);
  });

  // Delete handler
  container.querySelector('#char-delete-btn').addEventListener('click', () => {
    if (confirm(`Delete "${char.name}"? This cannot be undone.`)) {
      deleteCharacter(char.id);
    }
  });

  // ─────────────────────────────────────────────────────────────
  // Translate card — popover-driven, preview-first by default.
  // Backed by narrative_translate_default_language / _auto_save
  // user settings.
  // ─────────────────────────────────────────────────────────────
  const _FIELD_MAP = [
    ['#char-name-input', 'name', 'Name'],
    ['#char-desc', 'description', 'Description'],
    ['#char-personality', 'personality', 'Personality'],
    ['#char-scenario', 'scenario', 'Scenario'],
    ['#char-greeting', 'greeting', 'Greeting'],
    ['#char-examples', 'examples', 'Examples'],
    ['#char-creator-notes', 'creatorNotes', 'Creator notes'],
    ['#char-system-prompt', 'systemPrompt', 'System prompt'],
    ['#char-post-history', 'postHistoryInstructions', 'Post-history'],
    ['#char-depth-prompt', 'depthPrompt', 'Depth prompt'],
  ];

  const _collectFields = () => {
    const out = {};
    for (const [sel, key] of _FIELD_MAP) {
      out[key] = container.querySelector(sel)?.value || '';
    }
    return out;
  };

  const _applyTranslated = (translated, autosave) => {
    for (const [sel, key] of _FIELD_MAP) {
      const el = container.querySelector(sel);
      if (el && Object.prototype.hasOwnProperty.call(translated, key) && translated[key] != null) {
        el.value = translated[key];
      }
    }
    container.querySelectorAll('.field-textarea').forEach(ta => resizeNow(ta));
    if (autosave) {
      // Mirror the explicit save-button flow below so the translation
      // persists immediately. Without this, navigating away loses the
      // translation; users have repeatedly hit that.
      try {
        if (typeof syncFieldsToChar === 'function') syncFieldsToChar();
        saveCharacters(char.id);
        renderCharGrid();
        showToast('Translation applied and saved', 'success');
      } catch (err) {
        showToast(`Translation applied — save failed: ${err?.message || err}`, 'warning');
      }
    } else {
      showToast('Translation applied — click Save to persist', 'success');
    }
  };

  const _renderPreviewModal = (source, translated, autosave) => {
    const accepted = new Set(
      Object.keys(translated).filter(k => (translated[k] || '').trim() && (translated[k] !== source[k]))
    );
    const overlay = document.createElement('div');
    overlay.className = 'char-translate-preview-overlay';
    overlay.innerHTML = `
      <div class="char-translate-preview-modal" role="dialog" aria-label="Translation preview">
        <div class="char-translate-preview-header">
          <div class="char-translate-preview-title">Review translation</div>
          <div class="char-translate-preview-sub">${accepted.size} of ${_FIELD_MAP.length} fields changed</div>
        </div>
        <div class="char-translate-preview-body">
          ${_FIELD_MAP.map(([, key, label]) => {
            const before = source[key] || '';
            const after = translated[key] || '';
            const unchanged = before.trim() === after.trim();
            const empty = !before.trim() && !after.trim();
            const previewClass = empty
              ? 'char-translate-preview-row empty'
              : unchanged
                ? 'char-translate-preview-row unchanged'
                : 'char-translate-preview-row';
            return `
              <div class="${previewClass}" data-key="${escapeHtml(key)}">
                <label class="char-translate-preview-toggle">
                  <input type="checkbox" data-accept="${escapeHtml(key)}" ${accepted.has(key) ? 'checked' : ''} ${empty || unchanged ? 'disabled' : ''}>
                  <span class="char-translate-preview-label">${escapeHtml(label)}</span>
                  ${unchanged && !empty ? '<span class="char-translate-preview-tag">unchanged</span>' : ''}
                  ${empty ? '<span class="char-translate-preview-tag">empty</span>' : ''}
                </label>
                ${empty ? '' : `
                  <div class="char-translate-preview-cols">
                    <pre class="char-translate-preview-col before">${escapeHtml(before)}</pre>
                    <pre class="char-translate-preview-col after">${escapeHtml(after)}</pre>
                  </div>`}
              </div>`;
          }).join('')}
        </div>
        <div class="char-translate-preview-actions">
          <button type="button" class="btn btn-sm" id="char-translate-preview-cancel">Cancel</button>
          <button type="button" class="btn btn-sm" id="char-translate-preview-none">Select none</button>
          <button type="button" class="btn btn-sm" id="char-translate-preview-all">Select all</button>
          <button type="button" class="btn btn-primary btn-sm" id="char-translate-preview-apply">Apply selected</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    const close = () => overlay.remove();
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) close();
    });
    const escHandler = (e) => { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', escHandler); } };
    document.addEventListener('keydown', escHandler);

    overlay.querySelector('#char-translate-preview-cancel').addEventListener('click', close);
    overlay.querySelectorAll('input[data-accept]').forEach(cb => {
      cb.addEventListener('change', () => {
        const key = cb.dataset.accept;
        if (cb.checked) accepted.add(key); else accepted.delete(key);
      });
    });
    overlay.querySelector('#char-translate-preview-all').addEventListener('click', () => {
      overlay.querySelectorAll('input[data-accept]:not(:disabled)').forEach(cb => {
        cb.checked = true;
        accepted.add(cb.dataset.accept);
      });
    });
    overlay.querySelector('#char-translate-preview-none').addEventListener('click', () => {
      overlay.querySelectorAll('input[data-accept]').forEach(cb => {
        cb.checked = false;
        accepted.delete(cb.dataset.accept);
      });
    });
    overlay.querySelector('#char-translate-preview-apply').addEventListener('click', () => {
      const filtered = {};
      for (const key of accepted) {
        if (translated[key] != null) filtered[key] = translated[key];
      }
      if (Object.keys(filtered).length === 0) {
        showToast('No fields selected', 'warning');
        return;
      }
      close();
      document.removeEventListener('keydown', escHandler);
      _applyTranslated(filtered, autosave);
    });
  };

  // Popover wiring
  const translateBtn = container.querySelector('#char-translate-btn');
  const translatePopover = container.querySelector('#char-translate-popover');
  const translateTarget = container.querySelector('#char-translate-target');
  const translateCustom = container.querySelector('#char-translate-custom');
  const translateSource = container.querySelector('#char-translate-source');
  const translatePreview = container.querySelector('#char-translate-preview');
  const translateGo = container.querySelector('#char-translate-go');
  const translateCancel = container.querySelector('#char-translate-cancel');

  const _settingsForTranslate = () => {
    try { return getSettings() || {}; } catch { return {}; }
  };
  const _seedPopover = () => {
    const s = _settingsForTranslate();
    const defaultLang = s.narrativeTranslateDefaultLanguage || 'English';
    const knownOption = Array.from(translateTarget.options).find(o => o.value === defaultLang);
    if (knownOption) {
      translateTarget.value = defaultLang;
      translateCustom.classList.add('hidden');
      translateCustom.value = '';
    } else {
      translateTarget.value = '__custom__';
      translateCustom.classList.remove('hidden');
      translateCustom.value = defaultLang;
    }
    translatePreview.checked = true;  // preview-first default
  };

  translateTarget.addEventListener('change', () => {
    if (translateTarget.value === '__custom__') {
      translateCustom.classList.remove('hidden');
      translateCustom.focus();
    } else {
      translateCustom.classList.add('hidden');
    }
  });

  translateBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const wasHidden = translatePopover.classList.contains('hidden');
    translatePopover.classList.toggle('hidden');
    if (wasHidden) {
      _seedPopover();
      translateTarget.focus();
    }
  });

  const _outsideClose = (e) => {
    if (!translatePopover.classList.contains('hidden') &&
        !e.target.closest('.char-translate-popover') &&
        !e.target.closest('#char-translate-btn')) {
      translatePopover.classList.add('hidden');
    }
  };
  document.addEventListener('click', _outsideClose);

  translateCancel.addEventListener('click', () => {
    translatePopover.classList.add('hidden');
  });

  translateGo.addEventListener('click', async () => {
    const targetLang = translateTarget.value === '__custom__'
      ? (translateCustom.value || '').trim()
      : translateTarget.value;
    if (!targetLang) {
      showToast('Pick or type a target language', 'warning');
      translateCustom.focus();
      return;
    }
    const sourceLang = (translateSource.value || '').trim();
    const preview = !!translatePreview.checked;
    const fields = _collectFields();
    const nonEmpty = Object.values(fields).filter(v => (v || '').trim()).length;
    if (nonEmpty === 0) {
      showToast('Card has no text to translate', 'warning');
      return;
    }

    translatePopover.classList.add('hidden');
    const origText = translateBtn.textContent;
    translateBtn.textContent = 'Translating…';
    translateBtn.disabled = true;
    try {
      const activeModel = document.querySelector('#model-select')?.value || '';
      const resp = await fetch('/api/ui/translate-card', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fields,
          target_language: targetLang,
          source_language: sourceLang,
          model: activeModel,
          preview,
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (data.error) throw new Error(data.error);

      const settingsNow = _settingsForTranslate();
      const autosave = settingsNow.narrativeTranslateAutoSave !== false;

      if (preview) {
        _renderPreviewModal(data.source || fields, data.translated || {}, autosave);
      } else {
        _applyTranslated(data.translated || {}, autosave);
      }
    } catch (err) {
      showToast(`Translation failed: ${err.message}`, 'error');
    } finally {
      translateBtn.textContent = origText;
      translateBtn.disabled = false;
    }
  });

  // Add alternate greeting button
  container.querySelector('#add-alt-greeting-btn')?.addEventListener('click', () => {
    char.alternateGreetings = char.alternateGreetings || [];
    char.alternateGreetings.push('');
    renderCardEditor(char);
    // Focus the new textarea — save deferred until user types content
    setTimeout(() => {
      const texts = container.querySelectorAll('.alt-greeting-text');
      if (texts.length > 0) texts[texts.length - 1].focus();
    }, 50);
  });

  // Delegated events for alternate greetings
  container.addEventListener('click', (e) => {
    const useBtn = e.target.closest('.use-greeting-btn');
    if (useBtn) {
      const idx = parseInt(useBtn.dataset.idx);
      if (char.alternateGreetings?.[idx]) {
        const old = char.greeting || '';
        char.greeting = char.alternateGreetings[idx];
        char.alternateGreetings[idx] = old;
        if (!old) char.alternateGreetings.splice(idx, 1);
        saveCharacters();
        renderCardEditor(char);
        showToast('Greeting swapped', 'success');
      }
      return;
    }

    const delBtn = e.target.closest('.del-greeting-btn');
    if (delBtn) {
      const idx = parseInt(delBtn.dataset.idx);
      char.alternateGreetings?.splice(idx, 1);
      saveCharacters();
      renderCardEditor(char);
      return;
    }
  });
}

// ---------------------------------------------------------------------------
// Voice Options Loader
// ---------------------------------------------------------------------------

let _voiceListCache = null;

// Clear local voice cache when the central cache refreshes (provider CRUD).
onCacheChange('voices', () => { _voiceListCache = null; });

async function _loadVoiceOptions(selectEl, currentVoice) {
  // Populate from cache immediately if available
  if (_voiceListCache) {
    _populateVoiceSelect(selectEl, _voiceListCache, currentVoice);
    return;
  }

  try {
    const voices = await getVoices();
    _voiceListCache = voices;
    _populateVoiceSelect(selectEl, voices, currentVoice);
  } catch {
    // No TTS provider — leave default option only
  }
}

function _populateVoiceSelect(selectEl, voices, currentVoice) {
  // Keep the default option
  selectEl.innerHTML = '<option value="">Default (provider default)</option>';

  // Group voices by provider
  const byProvider = {};
  for (const v of voices) {
    const rawId = typeof v === 'string' ? v : (v.id || v.name || v.voice_id || '');
    if (!rawId) continue;
    const provId = (typeof v === 'object' && v.provider_id) ? v.provider_id : '';
    const provName = (typeof v === 'object' && v.provider_name) ? v.provider_name : '';
    // formatVoiceLabel adds the fabric source badge as text suffix when
    // applicable ("af_heart • 2"). Falls back to the bare name when the
    // entry is a plain string (legacy callers) or has no source metadata.
    const baseLabel = typeof v === 'string' ? v : (v.name || v.id || v.voice_id || rawId);
    const label = typeof v === 'object' ? formatVoiceLabel(v) : baseLabel;
    // Encode provider into value so TTS can route to the right backend
    const value = provId ? `${provId}::${rawId}` : rawId;
    const groupKey = provName || provId || 'default';
    if (!byProvider[groupKey]) byProvider[groupKey] = [];
    byProvider[groupKey].push({ value, label });
  }

  for (const [groupName, items] of Object.entries(byProvider)) {
    const group = document.createElement('optgroup');
    group.label = groupName;
    for (const { value, label } of items) {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = label;
      if (value === currentVoice) opt.selected = true;
      group.appendChild(opt);
    }
    selectEl.appendChild(group);
  }
}

// ---------------------------------------------------------------------------
// Start Chat with Character (greeting auto-send)
// ---------------------------------------------------------------------------

function startChatWithCharacter(char) {
  // Collect all available greetings
  const greetings = [];
  if (char.greeting) greetings.push(char.greeting);
  if (char.alternateGreetings?.length > 0) {
    for (const g of char.alternateGreetings) {
      if (g && !greetings.includes(g)) greetings.push(g);
    }
  }

  // If multiple greetings, show picker; otherwise start immediately
  if (greetings.length > 1) {
    openGreetingPicker(char, greetings);
  } else {
    launchChat(char, greetings[0] || `*${char.name} appears before you.*`);
  }
}

/** Expand common template macros in text ({{char}}, {{user}}, {{persona}}). */
function expandCardMacros(text, charName, userName, personaDesc = '') {
  if (!text || !text.includes('{{')) return text;
  return text
    .replace(/\{\{char\}\}/gi, charName || 'Character')
    .replace(/\{\{user\}\}/gi, userName || 'User')
    .replace(/\{\{obj\}\}/gi, userName || 'User')
    .replace(/\{\{persona\}\}/gi, personaDesc || '')
    .replace(/\{\{time\}\}/gi, new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
    .replace(/\{\{date\}\}/gi, new Date().toISOString().slice(0, 10))
    .replace(/\{\{day\}\}/gi, new Date().toLocaleDateString([], { weekday: 'long' }))
    .replace(/\{\{random\}\}/gi, () => Math.random().toFixed(4))
    .replace(/\{\{roll:(\d{1,3})d(\d{1,4})\}\}/gi, (_m, n, s) => {
      let total = 0;
      for (let i = 0; i < Math.min(parseInt(n), 100); i++) total += Math.floor(Math.random() * parseInt(s)) + 1;
      return String(total);
    });
}

function launchChat(char, greeting) {
  applyCharBackground(char);

  // Get the active persona for macro expansion and context injection
  const activePersona = personas.find(p => p.is_default);
  const userName = activePersona?.name || 'User';
  const charName = char.name || 'Character';
  const personaDesc = activePersona?.description || '';

  const systemParts = [];
  // Card system_prompt first (if set), then description
  if (char.systemPrompt) systemParts.push(expandCardMacros(char.systemPrompt, charName, userName, personaDesc));
  if (char.visualTraits) systemParts.push(`[Visual Traits]\n${char.visualTraits}`);
  if (char.description) systemParts.push(expandCardMacros(char.description, charName, userName, personaDesc));
  if (char.personality) systemParts.push(`Personality: ${expandCardMacros(char.personality, charName, userName, personaDesc)}`);
  if (char.scenario) systemParts.push(`Scenario: ${expandCardMacros(char.scenario, charName, userName, personaDesc)}`);

  // Inject user identity into context
  if (activePersona) {
    const personaParts = [];
    if (activePersona.name) personaParts.push(`Name: ${activePersona.name}`);
    if (activePersona.appearance) personaParts.push(`Appearance: ${activePersona.appearance}`);
    if (activePersona.description) personaParts.push(activePersona.description);
    if (personaParts.length > 0) {
      systemParts.push(`[User/Player Character]\n${personaParts.join('\n')}`);
    }
  }

  const systemPrompt = systemParts.join('\n\n');

  // Expand macros in greeting too
  const expandedGreeting = expandCardMacros(greeting, charName, userName, personaDesc);

  // Expand macros in examples
  const expandedExamples = expandCardMacros(char.examples || '', charName, userName, personaDesc);

  const event = new CustomEvent('narrative-start-chat', {
    detail: {
      characterId: char.id,
      characterName: charName,
      personaName: userName,
      greeting: expandedGreeting,
      systemPrompt,
      examples: expandedExamples,
      creatorNotes: char.creatorNotes || '',
      lorebook: (char.lorebook || []).filter(e => e.enabled),
      characterAvatar: char.avatar || '',
      userAvatar: activePersona?.avatar || '',
      characterVoice: char.voice || '',
    },
  });
  document.dispatchEvent(event);
}

// ---------------------------------------------------------------------------
// Greeting Picker Modal
// ---------------------------------------------------------------------------

let greetingModalEl = null;

function openGreetingPicker(char, greetings) {
  if (!greetingModalEl) {
    greetingModalEl = document.createElement('div');
    greetingModalEl.className = 'persona-modal-overlay';
    greetingModalEl.innerHTML = `
      <div class="persona-modal greeting-picker-modal">
        <div class="persona-modal-header">
          <span class="persona-modal-title">Choose a Greeting</span>
          <button class="icon-btn small" id="greeting-picker-close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div class="persona-modal-body">
          <p style="color:var(--text-secondary);font-size:var(--text-sm);margin-bottom:var(--space-sm)">Select a greeting to start the conversation:</p>
          <div class="greeting-picker-list" id="greeting-picker-list"></div>
        </div>
      </div>
    `;
    document.body.appendChild(greetingModalEl);

    greetingModalEl.querySelector('#greeting-picker-close').addEventListener('click', closeGreetingPicker);
    greetingModalEl.addEventListener('click', (e) => {
      if (e.target === greetingModalEl) closeGreetingPicker();
    });
  }

  const list = greetingModalEl.querySelector('#greeting-picker-list');
  list.innerHTML = greetings.map((g, i) => {
    const preview = g.length > 200 ? g.slice(0, 200) + '…' : g;
    const label = i === 0 ? 'Default Greeting' : `Alternate ${i}`;
    return `
      <button class="greeting-picker-item" data-idx="${i}">
        <span class="greeting-picker-label">${escapeHtml(label)}</span>
        <span class="greeting-picker-preview">${escapeHtml(preview)}</span>
      </button>
    `;
  }).join('');

  // Wire click handlers
  list.querySelectorAll('.greeting-picker-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.dataset.idx);
      closeGreetingPicker();
      launchChat(char, greetings[idx]);
    });
  });

  greetingModalEl.style.display = '';
}

function closeGreetingPicker() {
  if (greetingModalEl) {
    greetingModalEl.style.display = 'none';
    const list = greetingModalEl.querySelector('#greeting-picker-list');
    if (list) list.innerHTML = '';
  }
}

// ---------------------------------------------------------------------------
// Lorebook (inspector panel — lore-tab)
// ---------------------------------------------------------------------------

function renderLorebook() {
  const container = document.getElementById('lore-tab');
  if (!container) return;

  const char = getCharacter(activeCharId);
  if (!char) {
    container.innerHTML = `
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
        </svg>
        <p>Select a character to manage their lorebook.</p>
      </div>`;
    return;
  }

  const cardLore = char.lorebook || [];
  // Merge in AI-authored session entries that aren't in the character card
  const session = chat.getActiveSession?.();
  const sessionLore = (session?.lorebook || []).filter(e =>
    (e.source === 'narrative_established' || e.source === 'llm_authored')
    && !cardLore.some(c => c.id === e.id)
  );
  const lore = [...cardLore, ...sessionLore];

  let html = `<div style="padding:var(--space-md)">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:var(--space-sm)">
      <span class="field-label" style="margin:0">${lore.length} Entries${sessionLore.length ? ` (${sessionLore.length} AI-created)` : ''}</span>
      <div style="display:flex;gap:var(--space-xs)">
        <button class="btn btn-sm" id="save-global-lore-btn" title="Save all entries as a global lorebook collection">Save as Global</button>
        <button class="btn btn-sm" id="import-global-lore-btn" title="Import a global lorebook collection">Import Global</button>
        <button class="btn btn-sm" id="add-lore-btn">Add Entry</button>
      </div>
    </div>`;

  lore.forEach((entry, idx) => {
    const isConstant = entry.constant || false;
    const stickyVal = entry.sticky_turns || entry.sticky || 0;
    const cooldownVal = entry.cooldown_turns || entry.cooldown || 0;
    const priorityVal = entry.priority ?? entry.order ?? 100;
    const position = entry.position || 'before_char';
    const depthVal = entry.injection_depth ?? entry.depth ?? 4;
    const roleVal = entry.injection_role || entry.role || 'system';
    const atDepthVisible = position === 'at_depth' ? '' : 'display:none';

    const isAiAuthored = entry.source === 'narrative_established' || entry.source === 'llm_authored';
    html += `
      <div class="lore-entry${isAiAuthored ? ' lore-ai-authored' : ''}" data-lore-idx="${idx}">
        <div class="lore-entry-header">
          <svg class="lore-entry-toggle" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
          <span class="lore-entry-name">${escapeHtml(entry.name || 'Untitled')}</span>
          <span class="lore-entry-keys">${escapeHtml((entry.keys || []).join(', '))}</span>
          ${isAiAuthored ? '<span class="lore-ai-badge" title="Created by the AI during this session">AI</span>' : ''}
          ${isConstant ? '<span style="font-size:var(--text-xs);color:var(--accent);margin-left:auto;margin-right:var(--space-sm)" title="Always active">CONST</span>' : ''}
          <div class="lore-entry-enabled ${entry.enabled !== false ? 'on' : ''}" data-toggle-lore="${idx}"></div>
        </div>
        <div class="lore-entry-body">
          <div class="field-group">
            <label class="field-label">Name</label>
            <input type="text" class="field-input lore-name" value="${escapeHtml(entry.name || '')}">
          </div>
          <div class="field-group">
            <label class="field-label">Keywords (comma-separated)</label>
            <input type="text" class="field-input lore-keys" value="${escapeHtml((entry.keys || []).join(', '))}">
          </div>
          <div class="field-group">
            <label class="field-label">Content</label>
            <textarea class="field-textarea lore-content" rows="3">${escapeHtml(entry.content || '')}</textarea>
          </div>

          <div class="lore-grid-2col">
            <div class="field-group">
              <label class="field-label">Priority</label>
              <input type="number" class="field-input lore-priority" value="${priorityVal}" min="0" max="999" title="Lower = higher priority">
            </div>
            <div class="field-group">
              <label class="field-label">Position</label>
              <select class="field-input lore-position">
                <option value="before_char" ${position === 'before_char' ? 'selected' : ''}>Before Character</option>
                <option value="after_char" ${position === 'after_char' ? 'selected' : ''}>After Character</option>
                <option value="at_depth" ${position === 'at_depth' ? 'selected' : ''}>At Depth</option>
              </select>
            </div>
          </div>

          <div class="lore-grid-2col lore-at-depth-fields" style="${atDepthVisible}">
            <div class="field-group">
              <label class="field-label" title="0 = appended after the latest message; N = N turns back from the end">Depth</label>
              <input type="number" class="field-input lore-depth" value="${depthVal}" min="0" max="99">
            </div>
            <div class="field-group">
              <label class="field-label" title="Message role for the injected content">Role</label>
              <select class="field-input lore-role">
                <option value="system" ${roleVal === 'system' ? 'selected' : ''}>System</option>
                <option value="user" ${roleVal === 'user' ? 'selected' : ''}>User</option>
                <option value="assistant" ${roleVal === 'assistant' ? 'selected' : ''}>Assistant</option>
              </select>
            </div>
          </div>

          <div class="lore-grid-3col">
            <div class="field-group">
              <label class="field-label">Sticky</label>
              <input type="number" class="field-input lore-sticky" value="${stickyVal}" min="0" max="99" title="Stay active for N turns after trigger">
            </div>
            <div class="field-group">
              <label class="field-label">Cooldown</label>
              <input type="number" class="field-input lore-cooldown" value="${cooldownVal}" min="0" max="99" title="Block for N turns after sticky expires">
            </div>
            <div class="field-group lore-constant-wrap">
              <label class="field-label">&nbsp;</label>
              <label style="display:flex;align-items:center;gap:var(--space-xs);font-size:var(--text-sm);cursor:pointer">
                <input type="checkbox" class="lore-constant" ${isConstant ? 'checked' : ''}> Constant
              </label>
            </div>
          </div>

          <div style="display:flex;gap:var(--space-sm);margin-top:var(--space-sm)">
            <button class="btn btn-sm btn-primary save-lore-btn" data-save-lore="${idx}">Save</button>
            <button class="btn btn-sm delete-lore-btn" data-delete-lore="${idx}" style="color:var(--error)">Delete</button>
          </div>
        </div>
      </div>`;
  });

  html += '</div>';
  container.innerHTML = html;

  // Auto-resize lore content textareas when entry is expanded
  container.querySelectorAll('.lore-entry-header').forEach(header => {
    header.addEventListener('click', () => {
      setTimeout(() => {
        const body = header.parentElement.querySelector('.lore-entry-body');
        if (body) wireAutoResize(body);
      }, 50);
    });
  });

  // Add entry button
  container.querySelector('#add-lore-btn')?.addEventListener('click', () => {
    char.lorebook = char.lorebook || [];
    char.lorebook.push({ name: '', keys: [], content: '', enabled: true, priority: 100, position: 'before_char', sticky_turns: 0, cooldown_turns: 0, constant: false, injection_depth: 4, injection_role: 'system' });
    saveCharacters();
    renderLorebook();
  });

  // Delegated events for lore entries — attach ONCE. #lore-tab persists
  // across renderLorebook() calls; re-adding the listener every render
  // stacked handlers, so a click toggled `.open` N times (even N = no-op).
  // That's why expand only "worked sometimes". char/session are resolved
  // fresh inside so the single persistent handler never goes stale.
  if (!container.dataset.loreDelegated) {
    container.dataset.loreDelegated = '1';
    container.addEventListener('click', (e) => {
    const char = getCharacter(activeCharId);
    if (!char) return;
    // Toggle collapse
    const header = e.target.closest('.lore-entry-header');
    if (header && !e.target.closest('.lore-entry-enabled')) {
      header.parentElement.classList.toggle('open');
      return;
    }

    // Helper: resolve a rendered lore index to its backing array + local index.
    // Indices 0..cardLore-1 → char.lorebook; beyond that → session.lorebook (AI entries).
    const _cardLen = (char.lorebook || []).length;
    const _sess = chat.getActiveSession?.();
    function _resolveLoreTarget(idx) {
      if (idx < _cardLen) return { arr: char.lorebook, localIdx: idx, isSession: false };
      const sessionEntries = (_sess?.lorebook || []).filter(e =>
        (e.source === 'narrative_established' || e.source === 'llm_authored')
        && !(char.lorebook || []).some(c => c.id === e.id)
      );
      const sIdx = idx - _cardLen;
      if (sIdx < sessionEntries.length) {
        const realIdx = (_sess?.lorebook || []).indexOf(sessionEntries[sIdx]);
        return { arr: _sess.lorebook, localIdx: realIdx, isSession: true };
      }
      return null;
    }

    // Toggle enabled
    const toggle = e.target.closest('[data-toggle-lore]');
    if (toggle) {
      const idx = parseInt(toggle.dataset.toggleLore);
      const t = _resolveLoreTarget(idx);
      if (t && t.arr[t.localIdx]) {
        t.arr[t.localIdx].enabled = !t.arr[t.localIdx].enabled;
        toggle.classList.toggle('on');
        if (!t.isSession) saveCharacters();
      }
      return;
    }

    // Save entry
    const saveBtn = e.target.closest('[data-save-lore]');
    if (saveBtn) {
      const idx = parseInt(saveBtn.dataset.saveLore);
      const entryEl = container.querySelectorAll('.lore-entry')[idx];
      const t = _resolveLoreTarget(idx);
      if (entryEl && t && t.arr[t.localIdx]) {
        const target = t.arr[t.localIdx];
        target.name = entryEl.querySelector('.lore-name').value.trim();
        target.keys = entryEl.querySelector('.lore-keys').value.split(',').map(k => k.trim()).filter(Boolean);
        target.content = entryEl.querySelector('.lore-content').value;
        target.priority = parseInt(entryEl.querySelector('.lore-priority').value) || 100;
        target.position = entryEl.querySelector('.lore-position').value;
        target.sticky_turns = parseInt(entryEl.querySelector('.lore-sticky').value) || 0;
        target.cooldown_turns = parseInt(entryEl.querySelector('.lore-cooldown').value) || 0;
        target.constant = entryEl.querySelector('.lore-constant').checked;
        const depthInput = entryEl.querySelector('.lore-depth');
        const roleInput = entryEl.querySelector('.lore-role');
        if (depthInput) {
          const d = parseInt(depthInput.value);
          target.injection_depth = Number.isFinite(d) && d >= 0 ? d : 4;
        }
        if (roleInput) target.injection_role = roleInput.value;
        if (!t.isSession) saveCharacters();
        renderLorebook();
        showToast('Lore entry saved', 'success');
      }
      return;
    }

    // Delete entry
    const deleteBtn = e.target.closest('[data-delete-lore]');
    if (deleteBtn) {
      const idx = parseInt(deleteBtn.dataset.deleteLore);
      const t = _resolveLoreTarget(idx);
      if (!t) return;
      const entry = t.arr[t.localIdx];
      const name = (entry?.name || entry?.keys?.[0] || '').trim();
      const label = name ? `"${name}"` : 'this lore entry';
      if (!confirm(`Delete ${label}? This cannot be undone.`)) return;
      t.arr.splice(t.localIdx, 1);
      if (!t.isSession) saveCharacters();
      renderLorebook();
      return;
    }
  });

  // Toggle visibility of at-depth fields when the position select changes.
  // Uses 'change' instead of 'click' so it fires on both mouse and keyboard edits.
  container.addEventListener('change', (e) => {
    const sel = e.target.closest('.lore-position');
    if (!sel) return;
    const body = sel.closest('.lore-entry-body');
    const depthFields = body?.querySelector('.lore-at-depth-fields');
    if (depthFields) {
      depthFields.style.display = sel.value === 'at_depth' ? '' : 'none';
    }
    });
  }

  // Save entire lorebook as a global collection
  container.querySelector('#save-global-lore-btn')?.addEventListener('click', () => {
    _saveAsGlobalCollection(char);
  });

  // Import a global collection
  container.querySelector('#import-global-lore-btn')?.addEventListener('click', () => {
    _openGlobalLorebookModal(char);
  });
}

// ---------------------------------------------------------------------------
// Global Lorebook Collections
// ---------------------------------------------------------------------------

async function _saveAsGlobalCollection(char) {
  const lore = char.lorebook || [];
  if (lore.length === 0) {
    showToast('No lorebook entries to save', 'warning');
    return;
  }

  const name = prompt('Collection name:', `${char.name || 'Character'} Lore`);
  if (!name || !name.trim()) return;

  try {
    const resp = await fetch('/api/narrative/lorebook/global', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name.trim(),
        description: `From ${char.name || 'character'} (${lore.length} entries)`,
        entries: lore.map(e => ({
          name: e.name || '',
          keys: e.keys || [],
          content: e.content || '',
          enabled: e.enabled !== false,
          priority: e.priority ?? 100,
          position: e.position || 'before_char',
          sticky_turns: e.sticky_turns || 0,
          cooldown_turns: e.cooldown_turns || 0,
          constant: e.constant || false,
        })),
      }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    showToast(`Saved "${name.trim()}" (${data.entries} entries) to Global Library`, 'success');
  } catch (err) {
    showToast('Failed to save: ' + err.message, 'error');
  }
}

let _globalLoreModalEl = null;

async function _openGlobalLorebookModal(char) {
  let collections = [];
  try {
    const resp = await fetch('/api/narrative/lorebook/global');
    if (resp.ok) {
      const data = await resp.json();
      collections = data.collections || [];
    }
  } catch { /* ignore */ }

  if (collections.length === 0) {
    showToast('No global lorebook collections. Use "Save as Global" to create one.', 'info');
    return;
  }

  if (_globalLoreModalEl) _globalLoreModalEl.remove();
  _globalLoreModalEl = document.createElement('div');
  _globalLoreModalEl.className = 'modal-overlay';
  _globalLoreModalEl.style.display = 'flex';

  const rows = collections.map(c => `
    <div class="global-lore-row" style="display:flex;align-items:center;gap:var(--space-sm);padding:var(--space-md);border-bottom:1px solid var(--border-light)">
      <div style="flex:1;min-width:0">
        <div style="font-weight:600;font-size:var(--text-sm)">${escapeHtml(c.name)}</div>
        <div style="font-size:var(--text-xs);color:var(--text-muted)">${c.entry_count} entries${c.description ? ' — ' + escapeHtml(c.description) : ''}</div>
      </div>
      <button class="btn btn-sm btn-primary" data-import-collection="${escapeHtml(c.id)}">Import</button>
      <button class="btn btn-sm" data-rename-collection="${escapeHtml(c.id)}" title="Rename">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
      </button>
      <button class="btn btn-sm" data-delete-collection="${escapeHtml(c.id)}" style="color:var(--error)" title="Delete">&times;</button>
    </div>`).join('');

  _globalLoreModalEl.innerHTML = `
    <div class="modal" style="max-width:500px;max-height:70vh;display:flex;flex-direction:column">
      <div class="modal-header" style="display:flex;align-items:center;justify-content:space-between;padding:var(--space-md)">
        <span style="font-weight:600">Global Lorebook Collections</span>
        <button class="icon-btn small" id="global-lore-close">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div style="overflow-y:auto;flex:1">${rows}</div>
    </div>`;

  document.body.appendChild(_globalLoreModalEl);

  const closeModal = () => { _globalLoreModalEl?.remove(); _globalLoreModalEl = null; };
  _globalLoreModalEl.querySelector('#global-lore-close').addEventListener('click', closeModal);
  _globalLoreModalEl.addEventListener('click', (e) => { if (e.target === _globalLoreModalEl) closeModal(); });

  _globalLoreModalEl.addEventListener('click', async (e) => {
    // Import entire collection into character
    const importBtn = e.target.closest('[data-import-collection]');
    if (importBtn) {
      const colId = importBtn.dataset.importCollection;
      try {
        const resp = await fetch(`/api/narrative/lorebook/global/${encodeURIComponent(colId)}`);
        if (!resp.ok) throw new Error('Failed to fetch collection');
        const data = await resp.json();
        const entries = data.entries || [];
        if (entries.length === 0) {
          showToast('Collection is empty', 'warning');
          return;
        }
        char.lorebook = char.lorebook || [];
        for (const entry of entries) {
          char.lorebook.push({
            name: entry.name || '',
            keys: [...(entry.keys || [])],
            content: entry.content || '',
            enabled: entry.enabled !== false,
            priority: entry.priority ?? 100,
            position: entry.position || 'before_char',
            sticky_turns: entry.sticky_turns || 0,
            cooldown_turns: entry.cooldown_turns || 0,
            constant: entry.constant || false,
          });
        }
        saveCharacters();
        renderLorebook();
        showToast(`Imported ${entries.length} entries from "${data.name}"`, 'success');
        closeModal();
      } catch (err) {
        showToast('Import failed: ' + err.message, 'error');
      }
      return;
    }

    // Rename collection
    const renameBtn = e.target.closest('[data-rename-collection]');
    if (renameBtn) {
      const colId = renameBtn.dataset.renameCollection;
      const col = collections.find(c => c.id === colId);
      const newName = prompt('Collection name:', col?.name || '');
      if (!newName || !newName.trim()) return;
      try {
        await fetch(`/api/narrative/lorebook/global/${encodeURIComponent(colId)}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: newName.trim(), description: col?.description || '' }),
        });
        if (col) col.name = newName.trim();
        const nameEl = renameBtn.closest('.global-lore-row')?.querySelector('div > div:first-child');
        if (nameEl) nameEl.textContent = newName.trim();
        showToast('Collection renamed', 'success');
      } catch { /* ignore */ }
      return;
    }

    // Delete collection
    const deleteBtn = e.target.closest('[data-delete-collection]');
    if (deleteBtn) {
      const colId = deleteBtn.dataset.deleteCollection;
      const col = collections.find(c => c.id === colId);
      const label = col?.name?.trim() ? `"${col.name.trim()}"` : 'this collection';
      if (!confirm(`Delete ${label}? This cannot be undone.`)) return;
      try {
        const resp = await fetch(`/api/narrative/lorebook/global/${encodeURIComponent(colId)}`, { method: 'DELETE' });
        if (resp.ok) {
          collections = collections.filter(c => c.id !== colId);
          deleteBtn.closest('.global-lore-row')?.remove();
          showToast('Collection deleted', 'info');
          if (collections.length === 0) closeModal();
        }
      } catch { /* ignore */ }
    }
  });
}

// ---------------------------------------------------------------------------
// Chat Import / Export
// ---------------------------------------------------------------------------

async function exportNarrativeChat(sessionId) {
  const sessions = chat.getSessions();
  let session = sessions?.[sessionId];
  if (!session) return;

  // Session may be a metadata stub (no tree) — fetch full data from server
  if (!session.tree) {
    try {
      const resp = await fetch(`/api/chats/${encodeURIComponent(sessionId)}`);
      if (resp.ok) {
        session = await resp.json();
        sessions[sessionId] = session;
      }
    } catch { /* fall through with stub */ }
  }
  if (!session.tree) {
    showToast('Could not load chat data for export', 'error');
    return;
  }

  // Find the character associated with this session (prefers characterId)
  const char = _charForSession(session);

  // Build a flat message list from the tree (active branch only)
  const messages = [];
  if (session.tree && session.activeLeafId) {
    // Walk from active leaf to root to get the active branch path
    const path = [];
    let nodeId = session.activeLeafId;
    while (nodeId) {
      const node = session.tree[nodeId];
      if (!node) break;
      path.unshift(node);
      nodeId = node.parentId;
    }
    for (const node of path) {
      messages.push({
        role: node.role,
        content: node.content,
        createdAt: node.createdAt,
        ...(node.augmentum ? { augmentum: node.augmentum } : {}),
      });
    }
  } else if (session.tree) {
    // No active leaf — export all nodes sorted by creation time
    const nodes = Object.values(session.tree).sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
    for (const node of nodes) {
      messages.push({
        role: node.role,
        content: node.content,
        createdAt: node.createdAt,
        ...(node.augmentum ? { augmentum: node.augmentum } : {}),
      });
    }
  }

  // Build character card snapshot (without large avatar to keep file lean)
  let characterCard = null;
  if (char) {
    characterCard = { ...char };
    // Keep avatar reference but truncate if huge (>100KB)
    if (characterCard.avatar && characterCard.avatar.length > 100_000) {
      characterCard.avatarTruncated = true;
      delete characterCard.avatar;
    }
  }

  const exportData = {
    format: 'augmentum_narrative_v1',
    exportedAt: new Date().toISOString(),
    session: {
      id: session.id,
      title: session.title,
      mode: session.mode,
      version: session.version,
      tree: session.tree,
      rootId: session.rootId,
      activeLeafId: session.activeLeafId,
      createdAt: session.createdAt,
    },
    messages,
    character: characterCard,
  };

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const safeName = (session.title || 'chat').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 50);
  a.download = `narrative_${safeName}_${new Date().toISOString().split('T')[0]}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showToast('Chat exported', 'success');
}

function importNarrativeChat() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json,.jsonl';
  input.addEventListener('change', () => {
    if (input.files?.[0]) processNarrativeChatImport(input.files[0]);
  });
  input.click();
}

function processNarrativeChatImport(file) {
  const reader = new FileReader();
  reader.onload = async (e) => {
    try {
      const json = JSON.parse(e.target.result);
      let imported = false;

      if (json.format === 'augmentum_narrative_v1') {
        // Native Augmentum narrative export
        imported = await importAugmentumNarrative(json);
      } else if (json.format === 'augmentum_chat_v2' && json.session) {
        // Generic Augmentum chat export — treat as narrative if mode matches
        const session = json.session;
        if (session.version === 2 && session.tree) {
          // Assign new ID to avoid collision
          const newId = 's_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
          session.id = newId;
          const sessions = chat.getSessions();
          sessions[newId] = session;
          chat.saveSessions(newId);
          chat.switchSession(newId);
          imported = true;
        }
      } else if (Array.isArray(json) || (json.messages && Array.isArray(json.messages))) {
        // Generic message array (JSONL-like or OpenAI format)
        imported = importGenericMessages(json);
      } else if (json.histories || json.chat) {
        // SillyTavern chat export format
        imported = importSillyTavernChat(json);
      } else {
        showToast('Unrecognized chat format', 'error');
        return;
      }

      if (imported) {
        renderCharGrid();
        renderRecentChats();
        showToast('Chat imported', 'success');
      }
    } catch (err) {
      showToast('Failed to parse chat file: ' + err.message, 'error');
    }
  };
  reader.readAsText(file);
}

async function importAugmentumNarrative(json) {
  const { session, character } = json;
  if (!session || !session.tree) return false;

  // Import character card if included and not already present
  if (character && character.name) {
    const existing = characters.find(c => c.name === character.name);
    if (!existing) {
      // Create the character from the exported card
      const newChar = createCharacter(character.name);
      // Merge fields from the exported card
      Object.assign(newChar, character, { id: newChar.id });
      saveCharacters();
      renderCharGrid();
    }
  }

  // Assign new ID to avoid collision with existing sessions
  const newId = 's_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const newSession = {
    ...session,
    id: newId,
    mode: 'narrative',
  };

  // Remap tree node IDs aren't needed — they're internal and won't collide with session IDs
  const sessions = chat.getSessions();
  sessions[newId] = newSession;
  chat.saveSessions(newId);
  chat.switchSession(newId);
  return true;
}

function importGenericMessages(json) {
  const messages = Array.isArray(json) ? json : json.messages;
  if (!messages || messages.length === 0) return false;

  const char = getCharacter(activeCharId);
  const charName = char?.name || 'Imported Chat';

  // Build a V2 session tree from flat messages
  const newId = 's_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const tree = {};
  let prevId = null;
  let rootId = null;

  for (const msg of messages) {
    const role = msg.role || (msg.is_user ? 'user' : 'assistant');
    const content = msg.content || msg.mes || msg.text || '';
    if (!content) continue;

    const nodeId = 'n_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
    tree[nodeId] = {
      id: nodeId,
      role,
      content,
      parentId: prevId,
      children: [],
      createdAt: msg.createdAt || msg.send_date || Date.now(),
    };

    if (prevId && tree[prevId]) {
      tree[prevId].children.push(nodeId);
    }
    if (!rootId) rootId = nodeId;
    prevId = nodeId;
  }

  const sessions = chat.getSessions();
  sessions[newId] = {
    id: newId,
    title: charName,
    mode: 'narrative',
    version: 2,
    tree,
    rootId,
    activeLeafId: prevId,
    createdAt: Date.now(),
  };
  chat.saveSessions(newId);
  chat.switchSession(newId);
  return true;
}

function importSillyTavernChat(json) {
  // SillyTavern exports as { chat_metadata, chat: [...messages] }
  // or { histories: { default: [...messages] } }
  let messages;
  if (json.chat && Array.isArray(json.chat)) {
    messages = json.chat;
  } else if (json.histories?.default) {
    messages = json.histories.default;
  } else {
    return false;
  }

  // SillyTavern messages: { name, is_user, mes, send_date, ... }
  const converted = messages
    .filter(m => m.mes || m.content)
    .map(m => ({
      role: m.is_user ? 'user' : 'assistant',
      content: m.mes || m.content,
      createdAt: m.send_date ? new Date(m.send_date).getTime() : Date.now(),
    }));

  if (converted.length === 0) return false;

  return importGenericMessages({ messages: converted });
}

// ---------------------------------------------------------------------------
// Character Import / Export
// ---------------------------------------------------------------------------

function exportCharacter(char) {
  // Export as TavernCard v2 compatible JSON
  const extensions = {
    augmentum: { voice: char.voice || '' },
  };
  if (char.depthPrompt) {
    extensions.depth_prompt = char.depthPrompt;
    extensions.depth_prompt_depth = char.depthPromptDepth ?? 4;
  }

  const cardData = {
    spec: 'chara_card_v2',
    spec_version: '2.0',
    data: {
      name: char.name,
      description: char.description,
      personality: char.personality,
      scenario: char.scenario,
      first_mes: char.greeting,
      alternate_greetings: char.alternateGreetings || [],
      mes_example: char.examples,
      system_prompt: char.systemPrompt || '',
      post_history_instructions: char.postHistoryInstructions || '',
      creator_notes: char.creatorNotes || '',
      tags: char.tags || [],
      visual_traits: char.visualTraits || '',
      image_style: char.imageStyle || '',
      extensions,
      character_book: char.lorebook?.length > 0 ? {
        entries: Object.fromEntries(char.lorebook.map((e, i) => [String(i), {
          keys: e.keys || [],
          content: e.content || '',
          enabled: e.enabled !== false,
          name: e.name || '',
          order: e.priority ?? i,
          position: e.position === 'after_char' ? 1 : 0,
          constant: e.constant || false,
          sticky: e.sticky_turns || 0,
          cooldown: e.cooldown_turns || 0,
        }]))
      } : undefined,
    },
  };

  const blob = new Blob([JSON.stringify(cardData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${char.name.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showToast('Character exported as JSON', 'success');
}

async function exportCharacterAsPng(char) {
  // Build the same V2 card JSON used by exportCharacter()
  const extensions = { augmentum: { voice: char.voice || '' } };
  if (char.depthPrompt) {
    extensions.depth_prompt = char.depthPrompt;
    extensions.depth_prompt_depth = char.depthPromptDepth ?? 4;
  }
  const cardData = {
    spec: 'chara_card_v2', spec_version: '2.0',
    data: {
      name: char.name, description: char.description,
      personality: char.personality, scenario: char.scenario,
      first_mes: char.greeting, alternate_greetings: char.alternateGreetings || [],
      mes_example: char.examples, system_prompt: char.systemPrompt || '',
      post_history_instructions: char.postHistoryInstructions || '',
      creator_notes: char.creatorNotes || '', tags: char.tags || [],
      visual_traits: char.visualTraits || '',
      image_style: char.imageStyle || '',
      extensions,
      character_book: char.lorebook?.length > 0 ? {
        entries: Object.fromEntries(char.lorebook.map((e, i) => [String(i), {
          keys: e.keys || [], content: e.content || '', enabled: e.enabled !== false,
          name: e.name || '', order: e.priority ?? i,
          position: e.position === 'after_char' ? 1 : 0,
          constant: e.constant || false, sticky: e.sticky_turns || 0,
          cooldown: e.cooldown_turns || 0,
        }]))
      } : undefined,
    },
  };

  // Build V3 variant for the ccv3 chunk
  const v3CardData = { ...cardData, spec: 'chara_card_v3', spec_version: '3.0' };

  // Helper: build base64-encoded payload for a tEXt chunk
  function _buildTextChunkData(keyword, jsonObj) {
    const jsonStr = JSON.stringify(jsonObj);
    const jsonBytes = new TextEncoder().encode(jsonStr);
    // btoa can overflow on large strings — chunk it
    let b64 = '';
    for (let i = 0; i < jsonBytes.length; i += 8192) {
      b64 += String.fromCharCode(...jsonBytes.subarray(i, i + 8192));
    }
    b64 = btoa(b64);
    const data = new Uint8Array(keyword.length + 1 + b64.length);
    for (let i = 0; i < keyword.length; i++) data[i] = keyword.charCodeAt(i);
    data[keyword.length] = 0;
    for (let i = 0; i < b64.length; i++) data[keyword.length + 1 + i] = b64.charCodeAt(i);
    return data;
  }

  // Build both V2 (chara) and V3 (ccv3) tEXt chunk payloads
  const charaData = _buildTextChunkData('chara', cardData);
  const ccv3Data = _buildTextChunkData('ccv3', v3CardData);

  // Get base PNG from avatar or generate a placeholder
  let pngBuf;
  if (char.avatar) {
    let resp;
    try { resp = await fetch(char.avatar); } catch { resp = null; }
    if (!resp || !resp.ok) { showToast('Failed to load avatar for export', 'error'); return; }
    const blob = await resp.blob();
    // Convert to PNG via canvas
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve; img.onerror = reject;
      img.src = URL.createObjectURL(blob);
    });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);
    URL.revokeObjectURL(img.src);
    const pngBlob = await new Promise(r => canvas.toBlob(r, 'image/png'));
    pngBuf = await pngBlob.arrayBuffer();
  } else {
    // 1x1 transparent placeholder PNG
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 256;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#2a2a3e';
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = '#888';
    ctx.font = 'bold 48px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(char.name?.[0] || '?', 128, 128);
    const pngBlob = await new Promise(r => canvas.toBlob(r, 'image/png'));
    pngBuf = await pngBlob.arrayBuffer();
  }

  // Insert tEXt chunk before IEND in the PNG
  const src = new Uint8Array(pngBuf);
  // Find IEND offset
  let iendOffset = -1;
  for (let i = src.length - 12; i >= 8; i--) {
    if (src[i + 4] === 0x49 && src[i + 5] === 0x45 &&
        src[i + 6] === 0x4E && src[i + 7] === 0x44) {
      iendOffset = i; break;
    }
  }
  if (iendOffset < 0) { showToast('PNG encoding error', 'error'); return; }

  // Helper: wrap tEXt payload into a full PNG chunk (length + type + data + CRC)
  function _buildPngChunk(textData) {
    const len = textData.length;
    const chunk = new Uint8Array(12 + len);
    const dv = new DataView(chunk.buffer);
    dv.setUint32(0, len);
    chunk[4] = 0x74; chunk[5] = 0x45; chunk[6] = 0x78; chunk[7] = 0x74; // "tEXt"
    chunk.set(textData, 8);
    const c = crc32(chunk.slice(4, 8 + len));
    dv.setUint32(8 + len, c);
    return chunk;
  }

  const charaChunk = _buildPngChunk(charaData);
  const ccv3Chunk = _buildPngChunk(ccv3Data);

  // Assemble: [before IEND] + chara chunk + ccv3 chunk + [IEND chunk]
  const iendChunk = src.slice(iendOffset);
  const before = src.slice(0, iendOffset);
  const out = new Uint8Array(before.length + charaChunk.length + ccv3Chunk.length + iendChunk.length);
  out.set(before, 0);
  out.set(charaChunk, before.length);
  out.set(ccv3Chunk, before.length + charaChunk.length);
  out.set(iendChunk, before.length + charaChunk.length + ccv3Chunk.length);

  const blob = new Blob([out], { type: 'image/png' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${char.name.replace(/[^a-zA-Z0-9_-]/g, '_')}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showToast('Character exported as PNG card', 'success');
}

// CRC32 for PNG chunk validation
function crc32(bytes) {
  let table = crc32._table;
  if (!table) {
    table = crc32._table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      table[n] = c;
    }
  }
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) crc = table[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

// ---------------------------------------------------------------------------
// PNG Character Card Parsing
// Extracts tEXt chunks with keyword "ccv3" (V3) or "chara" (V2/V1)
// Data is base64-encoded JSON inside the Latin-1 tEXt chunk value
// ---------------------------------------------------------------------------

async function extractCharaFromPng(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);

  // Verify PNG signature
  const PNG_SIG = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) {
    if (bytes[i] !== PNG_SIG[i]) throw new Error('Not a valid PNG file');
  }

  const textChunks = {};
  let offset = 8;

  const view = new DataView(arrayBuffer);

  while (offset + 8 <= bytes.length) {
    // Read chunk length (4 bytes big-endian, unsigned)
    const length = view.getUint32(offset);
    offset += 4;

    // Read chunk type (4 bytes ASCII)
    const type = String.fromCharCode(bytes[offset], bytes[offset + 1],
                                     bytes[offset + 2], bytes[offset + 3]);
    offset += 4;

    // Safety: bail if length would exceed buffer
    if (length > bytes.length - offset) break;

    if (type === 'tEXt' && length > 0) {
      // tEXt chunk: keyword + null byte + text
      const data = bytes.slice(offset, offset + length);
      let nullIdx = -1;
      for (let i = 0; i < data.length; i++) {
        if (data[i] === 0) { nullIdx = i; break; }
      }
      if (nullIdx > 0) {
        const dec = new TextDecoder('latin1');
        const keyword = dec.decode(data.slice(0, nullIdx)).toLowerCase();
        textChunks[keyword] = dec.decode(data.slice(nullIdx + 1));
      }
    }

    offset += length + 4; // skip data + CRC

    if (type === 'IEND') break;
  }

  // Prefer ccv3 (V3 format), fall back to chara (V2/V1)
  const raw = textChunks['ccv3'] || textChunks['chara'];
  if (!raw) return null;

  // Base64 decode to UTF-8 JSON string
  // Use Uint8Array + TextDecoder instead of fetch('data:...') to avoid CSP connect-src blocks
  const binaryStr = atob(raw);
  const bytes2 = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) bytes2[i] = binaryStr.charCodeAt(i);
  const decoded = new TextDecoder('utf-8').decode(bytes2);
  return JSON.parse(decoded);
}

/** Extract `{cardData, avatarDataUrl}` from a base64 binary character payload.
 *  Sniffs in order: leading `{` = bare JSON; PNG signature = character-PNG
 *  (the file IS the avatar); ZIP signature = charx (avatar in `assets/icon.*`
 *  or filename match). Returns `{cardData: null, avatarDataUrl: null}` if
 *  none match.
 *
 *  Lives at module scope so both the URL-paste path (importFromUrl) and the
 *  RisuRealm import path can call it. */
async function extractCardFromBinaryPayload(base64Data) {
  const result = { cardData: null, avatarDataUrl: null };
  try {
    const binaryStr = atob(base64Data);
    const rawBytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) rawBytes[i] = binaryStr.charCodeAt(i);

    // Bare JSON
    if (rawBytes[0] === 0x7B) {
      try {
        result.cardData = JSON.parse(new TextDecoder().decode(rawBytes));
        return result;
      } catch { /* not JSON */ }
    }

    // Character PNG
    if (rawBytes[0] === 0x89 && rawBytes[1] === 0x50) {
      try {
        const json = await extractCharaFromPng(rawBytes.buffer);
        if (json) {
          result.cardData = json;
          const blobUrl = URL.createObjectURL(new Blob([rawBytes], { type: 'image/png' }));
          try { result.avatarDataUrl = await resizeAvatar(blobUrl); }
          finally { URL.revokeObjectURL(blobUrl); }
          return result;
        }
      } catch { /* not a character PNG */ }
    }

    // ZIP / charx (scan for PK\x03\x04 to handle SFX preambles too)
    const { BlobReader, ZipReader, BlobWriter } = await import('https://cdn.jsdelivr.net/npm/@zip.js/zip.js@2/lib/zip-no-worker.min.js');
    let zipOffset = -1;
    for (let i = 0; i < rawBytes.length - 4; i++) {
      if (rawBytes[i] === 0x50 && rawBytes[i+1] === 0x4B
          && rawBytes[i+2] === 0x03 && rawBytes[i+3] === 0x04) {
        zipOffset = i;
        break;
      }
    }
    if (zipOffset < 0) {
      console.warn('[card-import] No ZIP/PNG/JSON signature in binary (' + rawBytes.length + ' bytes)');
      return result;
    }

    const zipBuf = zipOffset > 0 ? rawBytes.slice(zipOffset) : rawBytes;
    const zipReader = new ZipReader(new BlobReader(new Blob([zipBuf])));
    try {
      const entries = await zipReader.getEntries();
      const entryMap = new Map();
      for (const e of entries) entryMap.set(e.filename, e);

      for (const entry of entries) {
        if (entry.filename === 'card.json' || entry.filename.endsWith('/card.json')) {
          const blob = await entry.getData(new BlobWriter());
          result.cardData = JSON.parse(await blob.text());
          break;
        }
      }

      const assets = result.cardData?.data?.assets || result.cardData?.assets || [];
      const iconAsset = assets.find(a => a?.type === 'icon')
                      || assets.find(a => a?.name === 'main');
      if (iconAsset?.uri) {
        const assetPath = iconAsset.uri
          .replace(/^embedded?:\/\//, '')
          .replace(/^__asset:/, '');
        const assetEntry = entryMap.get(assetPath) || entryMap.get('assets/' + assetPath);
        if (assetEntry) {
          const blob = await assetEntry.getData(new BlobWriter());
          const blobUrl = URL.createObjectURL(blob);
          try { result.avatarDataUrl = await resizeAvatar(blobUrl); }
          finally { URL.revokeObjectURL(blobUrl); }
        }
      }
      if (!result.avatarDataUrl) {
        for (const entry of entries) {
          if (/icon|avatar|main\.(png|jpe?g|webp)$/i.test(entry.filename)) {
            const blob = await entry.getData(new BlobWriter());
            const blobUrl = URL.createObjectURL(blob);
            try { result.avatarDataUrl = await resizeAvatar(blobUrl); }
            finally { URL.revokeObjectURL(blobUrl); }
            break;
          }
        }
      }
    } finally {
      await zipReader.close();
    }
  } catch (err) {
    console.warn('[card-import] binary extraction failed:', err);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Card Format Normalization
// Handles: TavernCard V3, V2, V1, Gradio/Pygmalion, Chub API response
// ---------------------------------------------------------------------------

function normalizeCardData(json) {
  // V2/V3: has spec field and nested data
  if (json.spec && json.data) {
    const data = json.data;
    // V3 assets live at top level — carry into data for downstream use
    if (Array.isArray(json.assets) && !data.assets) {
      data.assets = json.assets;
    }
    return data;
  }

  // V2 without spec but with nested data
  if (json.data && (json.data.name || json.data.char_name)) {
    return json.data;
  }

  // Chub API response: { node: { definition: { ... } } } — handles users
  // pasting raw responses from api.chub.ai/api/characters/<full>?full=true
  if (json.node?.definition) {
    return json.node.definition;
  }

  // Chub API response (already-unwrapped): definition field at top level
  if (json.definition) {
    return json.definition;
  }

  // Gradio / Pygmalion format: uses char_name, char_persona, etc.
  if (json.char_name) {
    return {
      name: json.char_name,
      description: json.char_persona || '',
      personality: '',
      scenario: json.world_scenario || '',
      first_mes: json.char_greeting || '',
      mes_example: json.example_dialogue || '',
    };
  }

  // V1 flat format or direct fields
  if (json.name || json.ch_name) {
    return json;
  }

  return null;
}

/** Strip stray underscores from card text that break macro detection and clutter output.
 *  - `_{{user}}_` → `{{user}}`   (markdown italic wrapping macros)
 *  - `_{{char}}_` → `{{char}}`
 *  - Leading/trailing `_` around words used as italic markers
 *  Preserves intentional underscores in identifiers (snake_case). */
function cleanCardText(text) {
  if (!text) return text;

  let cleaned = text;

  // Strip HTML if present — convert <img> to markdown, drop <style>, strip tags
  if (cleaned.includes('<') && /<[a-z]/i.test(cleaned)) {
    // Remove <style>...</style> blocks (JanitorAI CSS)
    cleaned = cleaned.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');
    // Convert <img> tags to markdown images
    cleaned = cleaned.replace(
      /<img\s+[^>]*src=["']([^"']+)["'][^>]*>/gi,
      (_match, src) => {
        if (/^https?:\/\/|^data:image\//i.test(src)) {
          const altMatch = _match.match(/alt=["']([^"']*?)["']/i);
          const alt = (altMatch?.[1] || 'image').replace(/[[\]()]/g, '');
          return `\n![${alt}](${src})\n`;
        }
        return '';
      }
    );
    // Extract background-image from inline styles
    cleaned = cleaned.replace(
      /style="[^"]*background(?:-image)?\s*:\s*url\(\s*['"]?(https?:\/\/[^'")]+)['"]?\s*\)[^"]*"/gi,
      (_match, url) => `\n![image](${url})\n`
    );
    // Convert block tags to newlines, strip remaining tags
    cleaned = cleaned.replace(/<\/?(?:p|div|br|hr|li|tr|h[1-6]|figure|figcaption|section|article)\b[^>]*>/gi, '\n');
    cleaned = cleaned.replace(/<[^>]+>/g, '');
    // Decode HTML entities
    const ta = document.createElement('textarea');
    ta.innerHTML = cleaned;
    cleaned = ta.value;
    // Collapse whitespace
    cleaned = cleaned.replace(/[ \t]+/g, ' ');
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
    cleaned = cleaned.trim();
  }

  // Remove _ immediately before/after macro braces: _{{ or }}_ or _{{user}}_ etc
  cleaned = cleaned.replace(/_(\{\{[^}]+\}\})_?/g, '$1');
  cleaned = cleaned.replace(/(\{\{[^}]+\}\})_/g, '$1');
  // Remove markdown italic markers: _word_ → word (but not snake_case mid-word)
  // Only strip when _ is at word boundary with space/start/end
  cleaned = cleaned.replace(/(^|[\s(])\_((?:[^_\n]|\S_\S)+?)\_(?=[\s).,!?;:]|$)/gm, '$1$2');
  return cleaned;
}

function applyCardDataToCharacter(char, data) {
  char.name = data.name || data.ch_name || data.char_name || char.name;
  char.description = cleanCardText(data.description || data.char_persona || '');
  char.personality = cleanCardText(data.personality || data.tavern_personality || '');
  char.scenario = cleanCardText(data.scenario || data.world_scenario || '');
  char.greeting = cleanCardText(data.first_mes || data.first_message || data.greeting_message ||
                  data.char_greeting || '');
  char.examples = cleanCardText(data.mes_example || data.example_dialogue || data.example_dialogs || '');
  const rawNotes = data.creator_notes || '';
  // Strip placeholder text that some card editors leave in
  char.creatorNotes = /^Creator'?s?\s*notes?\s*go\s*here\.?$/i.test(rawNotes.trim()) ? '' : rawNotes;

  // Visual traits (Augmentum extension — physical descriptors for image generation)
  if (data.visual_traits) {
    char.visualTraits = data.visual_traits;
  } else if (data.extensions?.visual_traits) {
    char.visualTraits = data.extensions.visual_traits;
  }

  // Image style (Augmentum extension — art style for background generation)
  if (data.image_style) {
    char.imageStyle = data.image_style;
  } else if (data.extensions?.image_style) {
    char.imageStyle = data.extensions.image_style;
  }

  // System prompt — stored separately so it round-trips on export
  if (data.system_prompt) {
    char.systemPrompt = cleanCardText(data.system_prompt);
  }

  // Post-history instructions (V2 standard field)
  if (data.post_history_instructions) {
    char.postHistoryInstructions = cleanCardText(data.post_history_instructions);
  }

  // Tags
  if (data.tags?.length > 0) {
    char.tags = data.tags;
  }

  // Alternate greetings
  const altGreets = data.alternate_greetings
    // JanitorAI: initial_messages is an array where [0] = primary, rest = alternates
    || (Array.isArray(data.initial_messages) && data.initial_messages.length > 1
        ? data.initial_messages.slice(1) : null);
  if (altGreets?.length > 0) {
    char.alternateGreetings = altGreets
      .map(g => cleanCardText(typeof g === 'string' ? g : (g?.message || g?.content || '')))
      .filter(g => g && g.trim());
  }

  // Extensions
  const ext = data.extensions || {};

  // depth_prompt (V2.1 extension)
  if (ext.depth_prompt) {
    char.depthPrompt = ext.depth_prompt;
    char.depthPromptDepth = ext.depth_prompt_depth ?? 4;
  }

  // Augmentum extensions (voice, etc.)
  if (ext.augmentum?.voice) {
    char.voice = ext.augmentum.voice;
  }

  // RisuAI extensions — extract emotion sprites if available
  if (ext.risuai) {
    const risuAssets = ext.risuai.additionalAssets || ext.risuai.emotions || [];
    if (risuAssets.length > 0 && !char.risuSprites) {
      char.risuSprites = risuAssets.map(a => ({
        name: Array.isArray(a) ? a[0] : (a.name || ''),
        url: Array.isArray(a) ? a[1] : (a.uri || a.url || ''),
      })).filter(s => s.name && s.url);
    }
  }

  // V3 assets: extract background and icon
  if (Array.isArray(data.assets)) {
    for (const asset of data.assets) {
      if (!asset?.uri) continue;
      if (asset.type === 'background' && !char.backgroundImage
          && /^https?:\/\//.test(asset.uri)) {
        char.backgroundImage = asset.uri;
      }
    }
  }

  // Avatar URL (JanitorAI uses 'photo' or 'profile_image'; TavernCard uses 'avatar')
  let avatarUrl = data.avatar || data.photo || data.profile_image || data.avatar_url || '';
  // V3 icon asset fallback
  if (!avatarUrl && Array.isArray(data.assets)) {
    const iconAsset = data.assets.find(a => a?.type === 'icon' && /^https?:\/\//.test(a.uri || ''));
    if (iconAsset) avatarUrl = iconAsset.uri;
  }
  if (typeof avatarUrl === 'string' && avatarUrl.startsWith('http') && !char.avatar) {
    // Store the URL for async download after this sync function returns
    char._pendingAvatarUrl = avatarUrl;
  }

  // Import lorebook / character_book / embedded_lorebook
  const book = data.character_book || data.embedded_lorebook;
  if (book?.entries) {
    const entries = Array.isArray(book.entries) ? book.entries : Object.values(book.entries);
    char.lorebook = entries.map(e => ({
      name: e.name || e.comment || '',
      keys: e.keys || e.key || [],
      content: e.content || '',
      enabled: e.enabled !== false,
      priority: e.order ?? e.priority ?? 100,
      position: e.position === 1 ? 'after_char' : (e.position === 'at_depth' ? 'at_depth' : 'before_char'),
      constant: e.constant || false,
      sticky_turns: e.sticky ?? e.sticky_turns ?? 0,
      cooldown_turns: e.cooldown ?? e.cooldown_turns ?? 0,
    }));
  }
}

/** Fetch an avatar image URL via the server proxy and return as a data URL. */
async function _fetchAvatarFromUrl(url) {
  try {
    const resp = await fetch('/api/ui/fetch-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    if (!resp.ok) return null;
    const result = await resp.json();
    if (result.type === 'binary' && result.data) {
      const ct = result.content_type || 'image/jpeg';
      return `data:${ct};base64,${result.data}`;
    }
  } catch (err) {
    console.warn('[Augmentum] Avatar download failed:', err.message);
  }
  return null;
}

// ---------------------------------------------------------------------------
// File Import (JSON + PNG)
// ---------------------------------------------------------------------------

function importCharacterFromFile(file) {
  return new Promise((resolve) => {
    const ext = file.name.split('.').pop()?.toLowerCase();

    if (ext === 'png') {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const json = await extractCharaFromPng(e.target.result);
          if (!json) {
            showToast(`No character data in ${file.name}`, 'error');
            resolve(null);
            return;
          }
          const blob = new Blob([e.target.result], { type: 'image/png' });
          const blobUrl = URL.createObjectURL(blob);
          const avatar = await resizeAvatar(blobUrl);
          URL.revokeObjectURL(blobUrl);
          resolve(importFromParsedJson(json, avatar, true));
        } catch (err) {
          showToast(`Failed to parse ${file.name}: ${err.message}`, 'error');
          resolve(null);
        }
      };
      reader.readAsArrayBuffer(file);
    } else if (ext === 'charx') {
      // Charx files are ZIP archives containing card.json + optional avatar.
      // May be self-extracting (SFX) with a JPEG/PNG preamble before the ZIP data.
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const { BlobReader, ZipReader, BlobWriter } = await import('https://cdn.jsdelivr.net/npm/@zip.js/zip.js@2/lib/zip-no-worker.min.js');

          // Handle SFX archives — scan for ZIP signature PK\x03\x04
          let rawBuf = e.target.result;
          const rawBytes = new Uint8Array(rawBuf);
          let zipOffset = 0;
          for (let i = 0; i < Math.min(rawBytes.length, 65536); i++) {
            if (rawBytes[i] === 0x50 && rawBytes[i+1] === 0x4B &&
                rawBytes[i+2] === 0x03 && rawBytes[i+3] === 0x04) {
              zipOffset = i;
              break;
            }
          }
          if (zipOffset > 0) {
            rawBuf = rawBuf.slice(zipOffset);
          }

          const zipReader = new ZipReader(new BlobReader(new Blob([rawBuf])));
          const entries = await zipReader.getEntries();

          let cardJson = null;
          let avatarDataUrl = null;
          const entryMap = new Map();
          for (const entry of entries) entryMap.set(entry.filename, entry);

          // Extract card.json
          for (const entry of entries) {
            if (entry.filename === 'card.json' || entry.filename.endsWith('/card.json')) {
              const blob = await entry.getData(new BlobWriter());
              cardJson = JSON.parse(await blob.text());
              break;
            }
          }

          if (cardJson) {
            // Avatar: prefer metadata-driven (data.assets icon), fall back to filename matching
            const assets = cardJson?.data?.assets || cardJson?.assets || [];
            const iconAsset = assets.find(a => a.type === 'icon') || assets.find(a => a.name === 'main');
            if (iconAsset?.uri) {
              // Resolve embedded:// or embeded:// or __asset: URIs
              let assetPath = iconAsset.uri
                .replace(/^embedded?:\/\//, '')
                .replace(/^__asset:/, '');
              const assetEntry = entryMap.get(assetPath) || entryMap.get('assets/' + assetPath);
              if (assetEntry) {
                const blob = await assetEntry.getData(new BlobWriter());
                const blobUrl = URL.createObjectURL(blob);
                avatarDataUrl = await resizeAvatar(blobUrl);
                URL.revokeObjectURL(blobUrl);
              }
            }

            // Filename fallback if metadata didn't yield an avatar
            if (!avatarDataUrl) {
              for (const entry of entries) {
                if (entry.filename.match(/icon|avatar|main\.(png|jpe?g|webp)$/i)) {
                  const blob = await entry.getData(new BlobWriter());
                  const blobUrl = URL.createObjectURL(blob);
                  avatarDataUrl = await resizeAvatar(blobUrl);
                  URL.revokeObjectURL(blobUrl);
                  break;
                }
              }
            }
          }

          await zipReader.close();

          if (!cardJson) {
            showToast(`No card.json found in ${file.name}`, 'error');
            resolve(null);
            return;
          }

          // CharaCard V3: {spec, spec_version, data: {...}}
          const definition = cardJson?.data || cardJson;
          resolve(importFromParsedJson(definition, avatarDataUrl, true));
        } catch (err) {
          showToast(`Failed to parse ${file.name}: ${err.message}`, 'error');
          resolve(null);
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const text = e.target.result;
          let json;
          if (ext === 'yaml' || ext === 'yml') {
            // Dynamic import of a lightweight YAML parser
            try {
              const { parse: parseYaml } = await import('https://cdn.jsdelivr.net/npm/yaml@2/+esm');
              json = parseYaml(text);
            } catch {
              showToast(`YAML parsing failed for ${file.name}. Check file format.`, 'error');
              resolve(null);
              return;
            }
          } else {
            json = JSON.parse(text);
          }
          resolve(importFromParsedJson(json, null, true));
        } catch {
          showToast(`Failed to parse ${file.name}`, 'error');
          resolve(null);
        }
      };
      reader.readAsText(file);
    }
  });
}

function _importSummary(char) {
  const parts = [`Imported "${char.name}"`];
  const details = [];
  if (char.avatar) details.push('avatar');
  if (char.description) details.push('description');
  if (char.personality) details.push('personality');
  if (char.greeting) details.push('greeting');
  if (char.scenario) details.push('scenario');
  if (char.examples) details.push('examples');
  if (char.systemPrompt) details.push('system prompt');
  const loreCount = (char.lorebook || []).length;
  if (loreCount > 0) details.push(`${loreCount} lore entries`);
  const altCount = (char.alternateGreetings || []).filter(g => g && g.trim()).length;
  if (altCount > 0) details.push(`${altCount} alt greetings`);
  if (details.length > 0) parts.push(`(${details.join(', ')})`);
  return parts.join(' ');
}

function importFromParsedJson(json, avatarDataUrl, batch = false) {
  const data = normalizeCardData(json);
  if (!data) {
    showToast('Unrecognized character card format', 'error');
    return null;
  }

  // In batch mode, create character object without saving/rendering per-file
  let char;
  if (batch) {
    const id = 'ch_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
    char = {
      id, name: data.name || data.char_name || 'Imported',
      description: '', personality: '', scenario: '', greeting: '',
      alternateGreetings: [], examples: '', systemPrompt: '',
      postHistoryInstructions: '', depthPrompt: '', depthPromptDepth: 4,
      creatorNotes: '', tags: [], avatar: null, backgroundImage: null,
      voice: '', lorebook: [], createdAt: Date.now(),
    };
    characters.push(char);
  } else {
    char = createCharacter(data.name || data.char_name || 'Imported');
  }

  applyCardDataToCharacter(char, data);

  if (avatarDataUrl) {
    char.avatar = avatarDataUrl;
  }

  // Download avatar from URL if present and no embedded avatar was provided
  if (!char.avatar && char._pendingAvatarUrl) {
    const pendingUrl = char._pendingAvatarUrl;
    delete char._pendingAvatarUrl;
    _fetchAvatarFromUrl(pendingUrl).then(dataUrl => {
      if (dataUrl) {
        char.avatar = dataUrl;
        saveCharacters(char.id);
        renderCharGrid();
      }
    });
  } else {
    delete char._pendingAvatarUrl;
  }

  if (!batch) {
    saveCharacters();
    renderCharGrid();
    selectCharacter(char.id, { openInspector: true });
    showToast(_importSummary(char), 'success');
  }
  return char;
}

async function importMultipleFiles(files) {
  const validFiles = files.filter(f => {
    const ext = f.name.split('.').pop()?.toLowerCase();
    return ext === 'png' || ext === 'json' || ext === 'charx' || ext === 'yaml' || ext === 'yml';
  });

  if (validFiles.length === 0) {
    showToast('No valid character card files selected', 'warning');
    return;
  }

  if (validFiles.length === 1) {
    // Single file — use standard flow with immediate UI update
    const char = await importCharacterFromFile(validFiles[0]);
    if (char) {
      saveCharacters();
      renderCharGrid();
      selectCharacter(char.id, { openInspector: true });
      showToast(`Imported "${char.name}"`, 'success');
    }
    return;
  }

  // Batch import — process all, then single save + render
  showToast(`Importing ${validFiles.length} characters...`, 'info', 3000);
  const results = [];
  for (const file of validFiles) {
    const char = await importCharacterFromFile(file);
    if (char) results.push(char);
  }

  if (results.length > 0) {
    saveCharacters();
    renderCharGrid();
    selectCharacter(results[results.length - 1].id, { openInspector: true });
    showToast(`Imported ${results.length} character${results.length > 1 ? 's' : ''}`, 'success');
  }
}

// ---------------------------------------------------------------------------
// URL Import
// Direct paste works for: Chub.ai, CharacterHub.org, Venus.chub.ai (parseChubUrl),
// RisuRealm (parseRisuRealmUrl), JanitorAI + JannyAI (_parseJanitorCharacterId),
// AICharacterCards.com (generic /api/ui/fetch-url → download_card_image
// extractor in ui_routes.py), and any direct .json/.png/.charx URL.
// Backyard.ai's hub does NOT expose a public download — don't add it without
// also adding a __NEXT_DATA__ scraper for their tRPC state.
// ---------------------------------------------------------------------------

function parseChubUrl(url) {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace('www.', '');
    // venus.chub.ai is the same backend (api.chub.ai) under a different
    // frontend host — verified by hitting the API with a fullPath taken
    // from a venus character URL.
    if (host !== 'chub.ai' && host !== 'characterhub.org'
        && host !== 'venus.chub.ai') return null;

    const segments = parsed.pathname.split('/').filter(Boolean);

    // /characters/creator/name or just creator/name
    let idx = segments.indexOf('characters');
    if (idx >= 0 && segments.length > idx + 2) {
      return { type: 'character', creator: segments[idx + 1], name: segments[idx + 2] };
    }

    // /lorebooks/creator/name
    idx = segments.indexOf('lorebooks');
    if (idx >= 0 && segments.length > idx + 2) {
      return { type: 'lorebook', creator: segments[idx + 1], name: segments[idx + 2] };
    }

    // Bare path: /creator/name (assume character)
    if (segments.length >= 2) {
      return { type: 'character', creator: segments[0], name: segments[1] };
    }
  } catch { /* not a valid URL */ }
  return null;
}

/** Recognize RisuRealm character URLs:
 *    https://realm.risuai.net/character/<id>
 *    https://realm.risuai.net/api/v1/download/dynamic/<id>?...
 *  Returns the bare character ID (UUID-ish string) or null. */
function parseRisuRealmUrl(url) {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace('www.', '');
    if (host !== 'realm.risuai.net' && host !== 'risuai.net') return null;

    const segments = parsed.pathname.split('/').filter(Boolean);

    // /character/<id>
    let idx = segments.indexOf('character');
    if (idx >= 0 && segments.length > idx + 1) return segments[idx + 1];

    // /api/v1/download/dynamic/<id>
    idx = segments.indexOf('dynamic');
    if (idx >= 0 && segments.length > idx + 1) return segments[idx + 1];
  } catch { /* not a valid URL */ }
  return null;
}

async function importFromChub(creator, name) {
  showToast('Fetching from Chub.ai...', 'info', 5000);

  try {
    // Use backend proxy to avoid CORS
    const resp = await fetch('/api/ui/fetch-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: `https://api.chub.ai/api/characters/${creator}/${name}?full=true`,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    const result = await resp.json();
    let apiData;

    if (result.type === 'json') {
      apiData = result.data;
    } else if (result.type === 'text') {
      apiData = JSON.parse(result.data);
    } else {
      throw new Error('Unexpected response type from Chub API');
    }

    // Chub API structure: { node: { definition: { ... card fields ... } } }
    const node = apiData.node || apiData;
    const definition = node.definition || node;

    const data = normalizeCardData(definition) || definition;
    if (!data || (!data.name && !data.char_name)) {
      throw new Error('No character data found in Chub response');
    }

    const char = createCharacter(data.name || data.char_name || `${creator}/${name}`);
    applyCardDataToCharacter(char, data);

    // Try to get avatar URL
    const avatarUrl = node.max_res_url || node.avatar_url || node.fullPath;
    if (avatarUrl) {
      char.avatar = avatarUrl;
    }

    saveCharacters();
    renderCharGrid();
    selectCharacter(char.id, { openInspector: true });
    showToast(_importSummary(char) + ' from Chub.ai', 'success');
  } catch (err) {
    showToast('Chub import failed: ' + err.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// JanitorAI / JannyAI import
// JanitorAI blocks direct fetches, but JannyAI (jannyai.com) mirrors their
// character database and exposes a download API.  We extract the UUID from
// janitorai.com or jannyai.com URLs and use the JannyAI download endpoint.
// ---------------------------------------------------------------------------

function _parseJanitorCharacterId(url) {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace('www.', '');
    if (host !== 'janitorai.com' && host !== 'jannyai.com') return null;
    // URL patterns:
    //   janitorai.com/characters/UUID_slug
    //   jannyai.com/characters/UUID_slug
    //   jannyai.com/characters/search?id=UUID
    const idParam = parsed.searchParams.get('id');
    if (idParam) return idParam;
    const segments = parsed.pathname.split('/').filter(Boolean);
    const idx = segments.indexOf('characters');
    if (idx >= 0 && segments.length > idx + 1) {
      // Extract UUID from "UUID_slug" format
      const raw = segments[idx + 1];
      const uuidMatch = raw.match(/^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i);
      return uuidMatch ? uuidMatch[1] : null;
    }
  } catch { /* not a valid URL */ }
  return null;
}

async function importFromJannyAI(characterId) {
  showToast('Fetching from JannyAI...', 'info', 5000);
  try {
    // Step 1: Get download URL via JannyAI API
    const dlResp = await fetch('/api/ui/fetch-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: 'https://api.jannyai.com/api/v1/download',
        method: 'POST',
        body: JSON.stringify({ characterId }),
        headers: { 'Content-Type': 'application/json' },
      }),
    });
    if (!dlResp.ok) throw new Error(`Download API returned ${dlResp.status}`);
    const dlResult = await dlResp.json();
    const dlData = dlResult.type === 'json' ? dlResult.data : JSON.parse(dlResult.data);

    if (dlData.status !== 'ok' || !dlData.downloadUrl) {
      throw new Error(dlData.error || 'Character not found on JannyAI');
    }

    // Step 2: Fetch the actual PNG card from the download URL
    const pngResp = await fetch('/api/ui/fetch-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: dlData.downloadUrl }),
    });
    if (!pngResp.ok) throw new Error(`Failed to download card: ${pngResp.status}`);
    const pngResult = await pngResp.json();

    if (pngResult.type === 'binary') {
      const rawDataUrl = 'data:' + (pngResult.content_type || 'image/png') + ';base64,' + pngResult.data;
      const blobResp = await fetch(rawDataUrl);
      const arrayBuf = await blobResp.arrayBuffer();
      const json = await extractCharaFromPng(arrayBuf);
      if (!json) {
        showToast('No character data found in downloaded card', 'error');
        return;
      }
      const avatar = await resizeAvatar(rawDataUrl);
      importFromParsedJson(json, avatar);
    } else if (pngResult.type === 'json') {
      importFromParsedJson(pngResult.data);
    } else {
      try { importFromParsedJson(JSON.parse(pngResult.data)); }
      catch { showToast('Downloaded card did not contain valid character data', 'error'); }
    }
  } catch (err) {
    showToast('JannyAI import failed: ' + err.message, 'error');
  }
}

async function importFromUrl(url) {
  // Check for Chub.ai URLs
  const chubInfo = parseChubUrl(url);
  if (chubInfo && chubInfo.type === 'character') {
    return importFromChub(chubInfo.creator, chubInfo.name);
  }

  // Check for RisuRealm character URLs — route through the dedicated importer
  // so we get the x-risu-api-version header + embedded-asset extraction.
  const risuId = parseRisuRealmUrl(url);
  if (risuId) {
    return _importFromRisu(risuId, '');
  }

  // Check for JanitorAI / JannyAI URLs
  const jannyId = _parseJanitorCharacterId(url);
  if (jannyId) {
    return importFromJannyAI(jannyId);
  }

  // Generic URL: fetch through backend proxy
  showToast('Fetching character card...', 'info', 5000);

  try {
    const resp = await fetch('/api/ui/fetch-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    const result = await resp.json();

    if (result.type === 'binary') {
      // Could be a character PNG (the file IS the avatar), a charx ZIP, or a
      // bare JSON payload mis-typed as octet-stream. extractCardFromBinaryPayload
      // sniffs all three and returns the avatar bytes when present.
      const extracted = await extractCardFromBinaryPayload(result.data);
      if (!extracted?.cardData) {
        showToast('No character data found in download', 'error');
        return;
      }
      importFromParsedJson(extracted.cardData, extracted.avatarDataUrl || null);
    } else if (result.type === 'json') {
      importFromParsedJson(result.data);
    } else if (result.type === 'text') {
      try {
        importFromParsedJson(JSON.parse(result.data));
      } catch {
        showToast('URL did not return valid character data', 'error');
      }
    }
  } catch (err) {
    showToast('URL import failed: ' + err.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Import Dialog (modal with file + URL options)
// ---------------------------------------------------------------------------

let importModalEl = null;

function openImportDialog() {
  if (!importModalEl) {
    importModalEl = document.createElement('div');
    importModalEl.className = 'modal-overlay hidden';
    importModalEl.innerHTML = `
      <div class="modal" style="width:min(620px,95vw);max-height:85vh;display:flex;flex-direction:column">
        <div class="modal-header">
          <span class="modal-title">Import Character</span>
          <button class="icon-btn small" id="import-close-btn" title="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        <div class="import-tabs">
          <button class="import-tab active" data-tab="search">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            Search
          </button>
          <button class="import-tab" data-tab="url">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
            URL / File
          </button>
        </div>

        <div class="modal-body" style="flex:1;overflow-y:auto;min-height:0">
          <!-- Search Tab -->
          <div class="import-tab-content active" data-tab="search">
            <div class="import-source-tabs">
              <button class="import-source-tab active" data-source="chub">Chub.ai</button>
              <button class="import-source-tab" data-source="risu">RisuRealm</button>
            </div>

            <div style="display:flex;gap:var(--space-sm)">
              <input type="text" class="field-input" id="char-search-input"
                placeholder="Search characters or leave empty to browse...">
              <button class="btn btn-primary btn-sm" id="char-search-btn">Search</button>
            </div>

            <!-- Chub filters -->
            <div class="chub-search-filters" id="chub-filters">
              <select class="field-input chub-filter-select" id="chub-sort">
                <option value="download_count">Most Downloaded</option>
                <option value="star_count">Most Starred</option>
                <option value="rating">Highest Rated</option>
                <option value="last_activity_at">Trending / Active</option>
                <option value="created_at">Newest</option>
                <option value="default">Relevance</option>
              </select>
              <label class="chub-filter-check" title="SFW only (on by default). Uncheck to include NSFW results — most Chub characters are tagged NSFW.">
                <input type="checkbox" id="chub-sfw-only" checked> SFW Only
              </label>
            </div>

            <!-- RisuRealm filters -->
            <div class="chub-search-filters hidden" id="risu-filters">
              <select class="field-input chub-filter-select" id="risu-sort">
                <option value="recommended">Recommended</option>
                <option value="downloads">Most Downloaded</option>
                <option value="trending">Trending</option>
                <option value="latest">Newest</option>
              </select>
              <label class="chub-filter-check" title="SFW only (on by default). Uncheck to include NSFW results.">
                <input type="checkbox" id="risu-sfw-only" checked> SFW Only
              </label>
            </div>

            <div id="char-search-results" class="chub-search-results">
              <div class="chub-search-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" width="32" height="32"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                <span>Search characters online</span>
                <span style="font-size:var(--text-xs);color:var(--text-muted)">Type a name (or leave blank) and click <strong>Search</strong> to browse. This reaches an external site — nothing is sent until you do. SFW-only is on by default.</span>
                <span id="char-source-label" style="font-size:var(--text-xs);color:var(--text-muted)">Powered by Chub.ai</span>
              </div>
            </div>

            <div id="char-search-pagination" class="chub-search-pagination hidden">
              <button class="btn btn-sm" id="char-prev-btn" disabled>Previous</button>
              <span id="char-page-info" style="font-size:var(--text-xs);color:var(--text-muted)"></span>
              <button class="btn btn-sm" id="char-next-btn">Next</button>
            </div>
          </div>

          <!-- URL / File Tab -->
          <div class="import-tab-content" data-tab="url">
            <div class="field-group">
              <label class="field-label">From URL</label>
              <div style="display:flex;gap:var(--space-sm)">
                <input type="text" class="field-input" id="import-url-input"
                  placeholder="https://chub.ai/characters/... or any URL from the sources below">
                <button class="btn btn-primary btn-sm" id="import-url-btn">Import</button>
              </div>
              <div style="font-size:var(--text-xs);color:var(--text-muted);margin-top:var(--space-xs)">
                Paste a character URL from any supported site below, or a direct link to a .json/.png/.charx card
              </div>
            </div>

            <div class="import-quicklinks">
              <span style="font-size:var(--text-xs);color:var(--text-muted)">Paste a URL from:</span>
              <a href="https://chub.ai/characters" target="_blank" rel="noopener" class="import-quicklink" title="Chub.ai — paste any character URL">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                Chub.ai
              </a>
              <a href="https://characterhub.org" target="_blank" rel="noopener" class="import-quicklink" title="CharacterHub — same backend as Chub">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                CharacterHub
              </a>
              <a href="https://venus.chub.ai" target="_blank" rel="noopener" class="import-quicklink" title="Venus.chub.ai — Chub frontend alias">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                Venus.chub
              </a>
              <a href="https://realm.risuai.net" target="_blank" rel="noopener" class="import-quicklink" title="RisuRealm — paste any /character/<id> URL">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                RisuRealm
              </a>
              <a href="https://janitorai.com" target="_blank" rel="noopener" class="import-quicklink" title="JanitorAI — pasted URLs are fetched via the JannyAI mirror">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                JanitorAI
              </a>
              <a href="https://jannyai.com" target="_blank" rel="noopener" class="import-quicklink" title="JannyAI — JanitorAI mirror, paste any /characters/<uuid> URL">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                JannyAI
              </a>
              <a href="https://aicharactercards.com/character-cards/" target="_blank" rel="noopener" class="import-quicklink" title="AICharacterCards — paste any character page URL">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                AICharacterCards
              </a>
            </div>

            <div style="display:flex;align-items:center;gap:var(--space-md);margin:var(--space-lg) 0">
              <div style="flex:1;height:1px;background:var(--border)"></div>
              <span style="font-size:var(--text-xs);color:var(--text-muted)">OR</span>
              <div style="flex:1;height:1px;background:var(--border)"></div>
            </div>

            <div class="field-group">
              <label class="field-label">JanitorAI Importer (Userscript)</label>
              <div style="display:flex;align-items:center;gap:var(--space-sm);flex-wrap:wrap">
                <a href="/ui/augmentum-janitor-import.user.js" id="janitor-userscript-install" class="btn btn-sm btn-primary" style="white-space:nowrap;text-decoration:none" title="Install userscript (requires Tampermonkey)">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                  Install Userscript
                </a>
                <span style="font-size:var(--text-xs);color:var(--text-muted);flex:1;min-width:180px">
                  Requires <a href="https://www.tampermonkey.net/" target="_blank" rel="noopener" style="color:var(--accent-text)">Tampermonkey</a>.
                  Once installed, a <strong>&ldquo;Send to Augmentum&rdquo;</strong> button appears on every JanitorAI character page.
                  Fetches the full card (description, personality, greeting, scenario, examples) directly from JanitorAI's API.
                </span>
              </div>
            </div>

            <div style="display:flex;align-items:center;gap:var(--space-md);margin:var(--space-lg) 0">
              <div style="flex:1;height:1px;background:var(--border)"></div>
              <span style="font-size:var(--text-xs);color:var(--text-muted)">OR</span>
              <div style="flex:1;height:1px;background:var(--border)"></div>
            </div>

            <div class="field-group">
              <label class="field-label">Paste JSON</label>
              <textarea class="field-input" id="import-paste-json" rows="4"
                placeholder='Paste raw character JSON here (from dev tools, API responses, any platform)...'
                style="resize:vertical;font-family:var(--font-mono,monospace);font-size:var(--text-xs)"></textarea>
              <div style="display:flex;gap:var(--space-sm);margin-top:var(--space-xs);align-items:center">
                <button class="btn btn-primary btn-sm" id="import-paste-btn">Parse & Import</button>
                <span style="font-size:var(--text-xs);color:var(--text-muted)">Works with TavernCard V1/V2/V3, charx, JanitorAI, Chub, RisuAI, Pygmalion formats</span>
              </div>
              <details style="margin-top:var(--space-sm)">
                <summary style="font-size:var(--text-xs);color:var(--text-muted);cursor:pointer;user-select:none">
                  How to grab JSON from JanitorAI (no extensions needed)
                </summary>
                <ol style="font-size:var(--text-xs);color:var(--text-secondary);margin:var(--space-xs) 0 0 var(--space-md);padding:0;line-height:1.6">
                  <li>Open a character page on JanitorAI</li>
                  <li>Press <kbd style="background:var(--surface-active);padding:1px 4px;border-radius:3px;font-family:var(--font-mono,monospace)">F12</kbd> to open DevTools &rarr; <strong>Network</strong> tab</li>
                  <li>Refresh the page (<kbd style="background:var(--surface-active);padding:1px 4px;border-radius:3px;font-family:var(--font-mono,monospace)">F5</kbd>)</li>
                  <li>In the Network filter, type the character&rsquo;s UUID (the long ID in the URL)</li>
                  <li>Click the matching request &rarr; <strong>Response</strong> tab &rarr; right-click &rarr; <strong>Copy value</strong></li>
                  <li>Paste here and click <strong>Parse &amp; Import</strong></li>
                </ol>
              </details>
            </div>

            <div style="display:flex;align-items:center;gap:var(--space-md);margin:var(--space-lg) 0">
              <div style="flex:1;height:1px;background:var(--border)"></div>
              <span style="font-size:var(--text-xs);color:var(--text-muted)">OR</span>
              <div style="flex:1;height:1px;background:var(--border)"></div>
            </div>

            <div class="import-dropzone" id="import-dropzone">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
              <span>Drop files here or click to browse</span>
              <span style="font-size:var(--text-xs);color:var(--text-muted)">PNG, JSON, CHARX, or YAML character cards (multiple supported)</span>
              <input type="file" accept=".json,.png,.charx,.yaml,.yml" id="import-file-input" multiple>
            </div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(importModalEl);

    // Close button
    importModalEl.querySelector('#import-close-btn').addEventListener('click', closeImportDialog);
    importModalEl.addEventListener('click', (e) => {
      if (e.target === importModalEl) closeImportDialog();
    });

    // Tab switching
    importModalEl.querySelectorAll('.import-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        importModalEl.querySelectorAll('.import-tab').forEach(t => t.classList.remove('active'));
        importModalEl.querySelectorAll('.import-tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        importModalEl.querySelector(`.import-tab-content[data-tab="${tab.dataset.tab}"]`).classList.add('active');
        if (tab.dataset.tab === 'search') {
          setTimeout(() => importModalEl.querySelector('#char-search-input').focus(), 50);
        }
      });
    });

    // --- Search Tab (multi-source: Chub + RisuRealm) ---
    const searchInput = importModalEl.querySelector('#char-search-input');
    const searchBtn = importModalEl.querySelector('#char-search-btn');
    const resultsEl = importModalEl.querySelector('#char-search-results');
    const paginationEl = importModalEl.querySelector('#char-search-pagination');
    // Pagination is *upstream* — each UI page change re-fetches. Earlier code
    // grabbed one upstream page (max 40 chub / 30-60 risu) then sliced locally
    // in chunks of 12, capping browseable results at the size of one upstream
    // response.
    const CHUB_PER_PAGE = 24;
    let _searchSource = 'chub'; // 'chub' | 'risu'
    let _searchPage = 1;        // 1-indexed; passed straight to upstream `page`
    let _searchQuery = '';
    let _pageNodes = [];        // current page only (was _allNodes)
    let _totalCount = null;     // chub returns `count`; risu doesn't expose total
    let _hasNext = false;       // upstream reported more after this page
    let _isFetching = false;    // guard double-fetch from rapid clicks
    // Privacy default: do NOT hit Chub/RisuRealm on open. The first online
    // request only fires when the user explicitly clicks Search (or Enter).
    // Before that, changing tabs/sorts/filters updates state silently.
    let _hasSearched = false;
    // NSFW opt-in is consent-gated once per user, remembered server-side.
    let _nsfwConsented = false;
    fetch('/api/config/ui').then(r => r.ok ? r.json() : null).then(d => {
      if (d && (d.nsfw_search_consent === 'true' || d.nsfw_search_consent === true)) {
        _nsfwConsented = true;
      }
    }).catch(() => {});

    // Source tab switching
    importModalEl.querySelectorAll('.import-source-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        importModalEl.querySelectorAll('.import-source-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        _searchSource = tab.dataset.source;
        // Toggle filter visibility
        importModalEl.querySelector('#chub-filters').classList.toggle('hidden', _searchSource !== 'chub');
        importModalEl.querySelector('#risu-filters').classList.toggle('hidden', _searchSource !== 'risu');
        const sourceLabel = importModalEl.querySelector('#char-source-label');
        if (sourceLabel) sourceLabel.textContent =
          _searchSource === 'chub' ? 'Powered by Chub.ai' : 'Powered by RisuRealm';
        _refetchIfSearched();
      });
    });

    const _resetAndFetch = () => {
      _searchPage = 1;
      _pageNodes = [];
      _totalCount = null;
      _hasNext = false;
      _fetchResults();
    };

    // Only re-fetch when the user has already opted into an online search.
    // Keeps tab/sort/filter changes from silently reaching out to Chub/Risu
    // before the user has clicked Search even once.
    const _refetchIfSearched = () => { if (_hasSearched) _resetAndFetch(); };

    // Consent gate: the first time a user unchecks "SFW Only" (opting into
    // NSFW results) we confirm intent and remember it per-user. Family-filtered
    // accounts never reach this — their toggle is force-checked + hidden and
    // the server strips NSFW regardless — but the guard is belt-and-suspenders.
    const _onSfwToggle = async (cb) => {
      if (!cb.checked && !isFamilyFiltered() && !_nsfwConsented) {
        const ok = await showConfirm({
          title: 'Show NSFW characters?',
          message: 'Unchecking "SFW Only" includes adult / NSFW characters in '
            + 'online results. Only continue if you want to see that content — '
            + 'you can turn "SFW Only" back on at any time.',
          confirmLabel: 'Show NSFW',
          cancelLabel: 'Keep SFW only',
          variant: 'danger',
        });
        if (!ok) { cb.checked = true; return; }   // reverted — stay SFW
        _nsfwConsented = true;
        fetch('/api/config/ui', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nsfw_search_consent: 'true' }),
        }).catch(() => {});
      }
      _refetchIfSearched();
    };

    const doSearch = () => {
      _searchQuery = searchInput.value.trim();
      _hasSearched = true;   // the explicit online opt-in
      _resetAndFetch();
    };
    searchBtn.addEventListener('click', doSearch);
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') doSearch();
    });

    // Filter changes re-fetch ONLY after the first explicit search.
    importModalEl.querySelector('#chub-sort').addEventListener('change', _refetchIfSearched);
    importModalEl.querySelector('#chub-sfw-only').addEventListener('change', (e) => _onSfwToggle(e.target));
    importModalEl.querySelector('#risu-sort').addEventListener('change', _refetchIfSearched);
    importModalEl.querySelector('#risu-sfw-only').addEventListener('change', (e) => _onSfwToggle(e.target));

    // Family-filtered accounts: force SFW-only to true and hide the
    // toggle so the UI matches what the server actually enforces. The
    // server is the source of truth (see /api/ui/fetch-url + /api/ui/
    // risurealm/search), so this is UX hygiene only — toggling via
    // devtools doesn't change the upstream result.
    if (isFamilyFiltered()) {
      for (const sel of ['#chub-sfw-only', '#risu-sfw-only']) {
        const cb = importModalEl.querySelector(sel);
        if (cb) {
          cb.checked = true;
          const label = cb.closest('label');
          if (label) label.style.display = 'none';
        }
      }
    }

    // Pagination — each click re-fetches the next/prev upstream page.
    importModalEl.querySelector('#char-prev-btn').addEventListener('click', () => {
      if (_searchPage > 1 && !_isFetching) { _searchPage--; _fetchResults(); }
    });
    importModalEl.querySelector('#char-next-btn').addEventListener('click', () => {
      if (_hasNext && !_isFetching) { _searchPage++; _fetchResults(); }
    });

    // ---- Unified fetch dispatcher ----
    async function _fetchResults() {
      if (_searchSource === 'chub') {
        await _fetchChubResults();
      } else {
        await _fetchRisuResults();
      }
    }

    // ---- Chub.ai ----
    async function _fetchChubResults() {
      const sort = importModalEl.querySelector('#chub-sort').value;
      const sfwOnly = importModalEl.querySelector('#chub-sfw-only').checked;

      _isFetching = true;
      resultsEl.innerHTML = '<div class="chub-search-loading">Searching...</div>';
      paginationEl.classList.add('hidden');

      try {
        // Chub's /search reads query-string params on GET; the JSON POST body
        // is silently ignored. Pass `page` upstream so the user can browse the
        // entire catalog instead of just the first response page.
        const params = new URLSearchParams({
          sort,
          asc: 'false',
          first: String(CHUB_PER_PAGE),
          page: String(_searchPage),
          // SFW is upstream's default — only flip when the user wants NSFW.
          nsfw: sfwOnly ? 'false' : 'true',
          nsfl: sfwOnly ? 'false' : 'true',
        });
        if (_searchQuery) params.set('search', _searchQuery);
        const upstreamUrl = `https://api.chub.ai/search?${params.toString()}`;

        const resp = await fetch('/api/ui/fetch-url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: upstreamUrl }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const result = await resp.json();
        const apiData = result.type === 'json' ? result.data : JSON.parse(result.data);
        const data = apiData?.data || apiData || {};
        const rawNodes = data.nodes || [];
        // `count` is sometimes the upstream total, sometimes just rawNodes.length.
        // Use it only when it clearly exceeds this page.
        _totalCount = (typeof data.count === 'number' && data.count > rawNodes.length)
          ? data.count : null;
        _hasNext = rawNodes.length === CHUB_PER_PAGE
                || (_totalCount !== null && _searchPage * CHUB_PER_PAGE < _totalCount);
        _pageNodes = rawNodes.map(n => ({
          _source: 'chub',
          id: n.fullPath || '',
          name: n.name || n.fullPath?.split('/').pop() || 'Unknown',
          creator: n.fullPath?.split('/')[0] || '',
          image_url: n.avatar_url || '',
          stats: [
            n.starCount ? `${_formatCount(n.starCount)} \u2733` : '',
            n.n_favorites ? `${_formatCount(n.n_favorites)} \u2665` : '',
            n.nChats ? `${_formatCount(n.nChats)} \uD83D\uDCAC` : '',
            n.nTokens ? `${_formatCount(n.nTokens)} tok` : '',
          ].filter(Boolean),
          tags: (n.topics || []).slice(0, 5),
          fullPath: n.fullPath || '',
          viewUrl: `https://chub.ai/characters/${n.fullPath || ''}`,
        }));
        _renderPage();
      } catch (err) {
        resultsEl.innerHTML = `<div class="chub-search-empty"><span style="color:var(--error)">Search failed: ${_escHtml(err.message)}</span></div>`;
        paginationEl.classList.add('hidden');
      } finally {
        _isFetching = false;
      }
    }

    // ---- RisuRealm ----
    async function _fetchRisuResults() {
      const sort = importModalEl.querySelector('#risu-sort').value;
      const sfwOnly = importModalEl.querySelector('#risu-sfw-only').checked;

      _isFetching = true;
      resultsEl.innerHTML = '<div class="chub-search-loading">Searching...</div>';
      paginationEl.classList.add('hidden');

      try {
        const params = new URLSearchParams({ sort, nsfw: !sfwOnly });
        if (_searchQuery) params.set('q', _searchQuery);
        if (_searchPage > 1) params.set('page', String(_searchPage));

        const resp = await fetch(`/api/ui/risurealm/search?${params}`);
        if (!resp.ok) {
          // RisuRealm's `recommended` feed is a single algorithmic page \u2014
          // upstream returns 500 on `recommended` + page>1. Fall back instead
          // of showing a red error.
          if (resp.status >= 500 && _searchPage > 1) {
            _hasNext = false;
            _searchPage--;  // stay on the last good page
            _renderPage();
            showToast('No more results in this feed', 'info', 2500);
            return;
          }
          throw new Error(`HTTP ${resp.status}`);
        }
        const result = await resp.json();
        if (result.error) throw new Error(result.error);

        const cards = result.cards || [];
        // RisuRealm returns 30 cards/page by default (60 for `recommended`).
        // A short page = end of results.
        const RISU_PAGE_SIZE = sort === 'recommended' ? 60 : 30;
        _hasNext = cards.length === RISU_PAGE_SIZE && sort !== 'recommended';
        _totalCount = null;  // upstream doesn't expose total
        _pageNodes = cards.map(c => ({
          _source: 'risu',
          id: c.id || '',
          name: c.name || 'Unknown',
          creator: c.creator || '',
          image_url: c.image_url || '',
          description: c.description || '',
          stats: [
            c.downloads ? `${c.downloads} \u2B07` : '',
            c.has_lorebook ? '\uD83D\uDCD6 Lore' : '',
          ].filter(Boolean),
          tags: (c.tags || []).slice(0, 5),
          viewUrl: `https://realm.risuai.net/character/${c.id}`,
        }));
        _renderPage();
      } catch (err) {
        resultsEl.innerHTML = `<div class="chub-search-empty"><span style="color:var(--error)">Search failed: ${_escHtml(err.message)}</span></div>`;
        paginationEl.classList.add('hidden');
      } finally {
        _isFetching = false;
      }
    }

    // ---- Unified page renderer ----
    function _renderPage() {
      if (_pageNodes.length === 0) {
        const hint = _searchQuery
          ? `No characters found for "${_escHtml(_searchQuery)}"`
          : 'No characters found';
        resultsEl.innerHTML = `
          <div class="chub-search-empty">
            <span>${hint}</span>
            <span style="font-size:var(--text-xs);color:var(--text-muted)">Try different keywords or filters</span>
          </div>`;
        paginationEl.classList.add('hidden');
        return;
      }

      resultsEl.innerHTML = '';
      for (const node of _pageNodes) {
        const card = document.createElement('div');
        card.className = 'chub-card';
        card.innerHTML = `
          <div class="chub-card-avatar">
            ${node.image_url ? `<img src="${_escHtml(node.image_url)}" alt="" loading="lazy" referrerpolicy="no-referrer" crossorigin="anonymous">` : `<div class="chub-card-avatar-placeholder">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="24" height="24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            </div>`}
          </div>
          <div class="chub-card-body">
            <div class="chub-card-name">${_escHtml(node.name)}</div>
            <div class="chub-card-creator">by ${_escHtml(node.creator)}</div>
            ${node.description ? `<div class="chub-card-desc">${_escHtml(node.description.slice(0, 80))}${node.description.length > 80 ? '...' : ''}</div>` : ''}
            <div class="chub-card-stats">
              ${node.stats.map(s => `<span>${s}</span>`).join('')}
            </div>
            ${node.tags.length ? `<div class="chub-card-tags">${node.tags.map(t => `<span class="chub-tag">${_escHtml(t)}</span>`).join('')}</div>` : ''}
          </div>
          <div class="chub-card-actions">
            <a href="${_escHtml(node.viewUrl)}" target="_blank" rel="noopener noreferrer"
              class="btn btn-sm chub-card-view" title="View on ${node._source === 'chub' ? 'Chub.ai' : 'RisuRealm'}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
            </a>
            <button class="btn btn-primary btn-sm chub-card-import" title="Import this character">Import</button>
          </div>
        `;
        card.querySelector('.chub-card-import').addEventListener('click', (e) => {
          e.stopPropagation();
          if (node._source === 'chub') {
            const parts = node.fullPath.split('/');
            if (parts.length >= 2) {
              closeImportDialog();
              importFromChub(parts[0], parts.slice(1).join('/'));
            }
          } else {
            closeImportDialog();
            _importFromRisu(node.id, node.name);
          }
        });
        resultsEl.appendChild(card);
      }

      paginationEl.classList.remove('hidden');
      importModalEl.querySelector('#char-prev-btn').disabled = _searchPage <= 1;
      // Honest count: chub gives `count` (when paginated); risu doesn't expose
      // a total. Show "Page N" alone when total is unknown rather than lying
      // with the per-page count as a global total.
      let info;
      if (_totalCount !== null) {
        const totalPages = Math.ceil(_totalCount / CHUB_PER_PAGE);
        info = `Page ${_searchPage} of ${totalPages} (${_totalCount.toLocaleString()} results)`;
      } else {
        info = `Page ${_searchPage}`;
      }
      importModalEl.querySelector('#char-page-info').textContent = info;
      importModalEl.querySelector('#char-next-btn').disabled = !_hasNext;
    }

    // ---- RisuRealm import ----
    async function _importFromRisu(id, name) {
      showToast(`Fetching from RisuRealm...`, 'info', 5000);
      try {
        const resp = await fetch('/api/ui/fetch-url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: `https://realm.risuai.net/api/v1/download/dynamic/${id}?cors=true`,
            headers: { 'x-risu-api-version': '4' },
          }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const result = await resp.json();
        let cardData;
        let avatarDataUrl = null;
        if (result.type === 'json') {
          cardData = result.data;
        } else if (result.type === 'binary') {
          // Charx ZIP / character PNG — extract card.json + avatar client-side
          const extracted = await _extractCardFromBinaryCharx(result.data);
          cardData = extracted?.cardData;
          avatarDataUrl = extracted?.avatarDataUrl || null;
          if (!cardData) throw new Error('Could not extract card data from RisuRealm charx response');
        } else if (result.type === 'text' && typeof result.data === 'string') {
          try {
            cardData = JSON.parse(result.data);
          } catch {
            throw new Error('RisuRealm returned non-JSON data. The character may use a format we cannot parse.');
          }
        } else {
          throw new Error(`Unexpected response type: ${result.type}`);
        }

        // RisuRealm returns CharaCard V3 spec: {spec, spec_version, data: {...}}
        const definition = cardData?.data || cardData;
        const data = normalizeCardData(definition) || definition;
        if (!data || (!data.name && !data.char_name)) {
          console.warn('[RisuRealm] Card data keys:', Object.keys(cardData || {}));
          if (cardData?.data) console.warn('[RisuRealm] data keys:', Object.keys(cardData.data));
          throw new Error('No character data found in RisuRealm response');
        }

        console.debug('[RisuRealm] Importing card fields:', Object.keys(data).filter(k => data[k]));
        // RisuAI-native cards store description in extensions.risuai fields
        const risuai = data.extensions?.risuai;
        if (risuai) {
          // additionalText is RisuAI's description/system prompt equivalent
          if (risuai.additionalText && !data.description) {
            data.description = risuai.additionalText;
          }
          // newGenData may contain structured persona info
          if (risuai.newGenData && typeof risuai.newGenData === 'object') {
            if (risuai.newGenData.personality && !data.personality) {
              data.personality = risuai.newGenData.personality;
            }
            if (risuai.newGenData.scenario && !data.scenario) {
              data.scenario = risuai.newGenData.scenario;
            }
            if (risuai.newGenData.description && !data.description) {
              data.description = risuai.newGenData.description;
            }
          }
        }
        const char = createCharacter(data.name || data.char_name || name);
        applyCardDataToCharacter(char, data);
        // Embedded avatar (PNG body or charx asset) wins over a URL fallback —
        // it's already local, doesn't need a second proxy hop, and is what
        // the card's author actually packaged.
        if (avatarDataUrl) {
          char.avatar = avatarDataUrl;
          char._pendingAvatarUrl = null;
        }

        saveCharacters();
        renderCharGrid();
        selectCharacter(char.id, { openInspector: true });
        showToast(_importSummary(char) + ' from RisuRealm', 'success');
      } catch (err) {
        showToast(`Import failed: ${err.message}`, 'error');
      }
    }

    // Thin alias — implementation lives at module scope as
    // `extractCardFromBinaryPayload` so importFromUrl can reuse it.
    const _extractCardFromBinaryCharx = extractCardFromBinaryPayload;

    function _escHtml(s) {
      const d = document.createElement('div');
      d.textContent = s || '';
      return d.innerHTML;
    }

    function _formatCount(n) {
      if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
      if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
      return String(n);
    }

    // --- URL Tab ---
    importModalEl.querySelector('#import-url-btn').addEventListener('click', () => {
      const url = importModalEl.querySelector('#import-url-input').value.trim();
      if (!url) {
        showToast('Please enter a URL', 'warning');
        return;
      }
      closeImportDialog();
      importFromUrl(url);
    });

    importModalEl.querySelector('#import-url-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        importModalEl.querySelector('#import-url-btn').click();
      }
    });

    // --- JanitorAI Userscript ---
    // The userscript (.user.js) is served as a static file from /ui/
    // Tampermonkey detects the .user.js extension and prompts to install
    // No additional JS setup needed here — the install link is in the HTML above

    // --- Paste JSON ---
    importModalEl.querySelector('#import-paste-btn').addEventListener('click', () => {
      const raw = importModalEl.querySelector('#import-paste-json').value.trim();
      if (!raw) {
        showToast('Please paste some JSON first', 'warning');
        return;
      }
      let json;
      try {
        json = JSON.parse(raw);
      } catch {
        showToast('Invalid JSON — could not parse', 'error');
        return;
      }
      closeImportDialog();
      const result = importFromParsedJson(json);
      if (result) {
        importModalEl.querySelector('#import-paste-json').value = '';
      }
    });

    // Dropzone click
    const dropzone = importModalEl.querySelector('#import-dropzone');
    const fileInput = importModalEl.querySelector('#import-file-input');
    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
      const files = Array.from(fileInput.files || []);
      if (files.length > 0) {
        closeImportDialog();
        importMultipleFiles(files);
        fileInput.value = '';
      }
    });

    // Drag & drop
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      const files = Array.from(e.dataTransfer?.files || []);
      if (files.length > 0) {
        closeImportDialog();
        importMultipleFiles(files);
      }
    });

    // No auto-search on open — the character search reaches an external site
    // (Chub.ai / RisuRealm), so we wait for the user to click Search. The
    // empty state explains this; nothing leaves the machine until then.
  }

  importModalEl.querySelector('#import-url-input').value = '';
  importModalEl.classList.remove('hidden');
  setTimeout(() => importModalEl.querySelector('#char-search-input').focus(), 100);
}

function closeImportDialog() {
  if (importModalEl) importModalEl.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Narrative State Polling (server-side state from backend)
// ---------------------------------------------------------------------------

function _dispatchStateEvent(data, sessionId) {
  document.dispatchEvent(new CustomEvent('augmentum:narrative-state', {
    detail: { data, sessionId },
  }));
}

// Track which stale-model alert we've already shown so the 5s poll doesn't
// re-toast the same broken reference every tick. Keyed by requested model.
let _shownModelAlertFor = '';

// The memory (LTM) refresh model this card/setting points at is gone. Rather
// than let the server silently reroute to whatever the engine has loaded
// (never auto-select), surface the choice: skip, use the chat model, or pin
// the first engine model. Each is an explicit user pick.
async function _maybeShowModelAlert(alert, sessionId) {
  if (!alert || !alert.requested) {
    _shownModelAlertFor = '';
    return;
  }
  if (_shownModelAlertFor === alert.requested) return; // already offered
  _shownModelAlertFor = alert.requested;

  const requested = alert.requested;
  const send = async (action) => {
    try {
      const resp = await fetch(`/api/narrative/session/${sessionId}/resolve-model-alert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      return resp.ok ? await resp.json().catch(() => null) : null;
    } catch { return null; /* best-effort; alert re-fires next refresh if unresolved */ }
  };

  // Reflect the applied choice in the LTM dropdown + local state so the panel
  // stays truthful without waiting for the next poll. The endpoint echoes the
  // model it settled on (empty string = chat model).
  const applyLocal = (val) => {
    _ltmModel = val || '';
    const sel = document.getElementById('ns-ltm-model');
    if (sel) { try { sel.value = val || ''; } catch { /* option may be absent */ } }
  };

  showChoiceToast(
    `Memory model "${requested}" is no longer available`,
    [
      {
        label: 'Use chat model',
        primary: true,
        onClick: async () => { await send('use_chat_model'); applyLocal(''); _shownModelAlertFor = ''; },
      },
      {
        label: 'Use engine model',
        onClick: async () => {
          const res = await send('use_engine_model');
          applyLocal(res?.narrative_memory_model || '');
          _shownModelAlertFor = '';
        },
      },
      {
        label: 'Skip',
        onClick: async () => { await send('skip'); },
      },
    ],
    { description: 'The long-term memory refresh needs a model. Pick what it should use.', type: 'warning' },
  );
}

async function pollNarrativeState() {
  try {
    const sessionId = app.state.currentSessionId;

    // Try current session state first
    if (app.state.mode === 'narrative' && sessionId) {
      const [stateResp, archiveResp] = await Promise.all([
        fetch(`/api/ui/session/${sessionId}/state`),
        fetch(`/api/ui/session/${sessionId}/archive`).catch(() => null),
      ]);
      if (stateResp.ok) {
        const data = await stateResp.json();
        if (data.state) {
          // Attach archive data to state for the inspector
          if (archiveResp && archiveResp.ok) {
            const archData = await archiveResp.json();
            data.state._archive = archData.exchanges || [];
          } else {
            data.state._archive = [];
          }
          const fp = _stateFingerprint(data.state);
          const changed = fp !== _lastStateFingerprint;
          narrativeState = data.state;
          if (changed) {
            _lastStateFingerprint = fp;
            renderNarrativeStateInspector();
          }
          _maybeShowModelAlert(data.state.model_alert, sessionId);
          _dispatchStateEvent(data, null);
          pollNarrativeBackground(sessionId);
          return;
        }
      }

      // Current session has no state yet — show empty LTM panel.
      // Do NOT fall through to the server-side scan, which would pick
      // up a different character's state and display it here.
      const wasNull = narrativeState === null;
      narrativeState = null;
      _dispatchStateEvent({ mode: 'passthrough', state: null }, null);
      if (!wasNull) {
        _lastStateFingerprint = '';
        renderNarrativeStateInspector();
      }
      return;
    }

    // Fallback: no active session — scan server-side sessions.
    // This path exists for external clients (Open WebUI, SillyTavern)
    // that don't use the built-in chat UI session management.
    const sessResp = await fetch('/api/ui/sessions');
    if (sessResp.ok) {
      const sessData = await sessResp.json();
      if (sessData.sessions?.length > 0) {
        for (const sess of sessData.sessions) {
          const resp = await fetch(`/api/ui/session/${sess.session_id}/state`);
          if (resp.ok) {
            const data = await resp.json();
            if (data.state) {
              const fp = _stateFingerprint(data.state);
              const changed = fp !== _lastStateFingerprint;
              narrativeState = data.state;
              if (changed) {
                _lastStateFingerprint = fp;
                renderNarrativeStateInspector();
              }
              _dispatchStateEvent(data, sess.session_id);
              pollNarrativeBackground(sess.session_id);
              return;
            }
          }
        }
      }
    }

    const wasNull = narrativeState === null;
    narrativeState = null;
    _dispatchStateEvent({ mode: 'passthrough', state: null }, null);
    if (!wasNull) {
      _lastStateFingerprint = '';
      renderNarrativeStateInspector();
    }
  } catch { /* silently fail */ }
}

/** Build a lightweight fingerprint of the narrative state for change detection. */
function _stateFingerprint(state) {
  if (!state) return '';
  // Include all fields that drive visible UI changes
  const snapshotLen = JSON.stringify(state.state_snapshot || {}).length;
  const ledgerLen = (state.memory_ledger || []).length;
  const lastLedger = ledgerLen > 0 ? state.memory_ledger[ledgerLen - 1]?.content?.slice(-30) || '' : '';
  return `${state.message_count}|${(state.memory_summary || '').length}|${state.memory_summary?.slice(-40) || ''}|${JSON.stringify(state.relationships || []).length}|${snapshotLen}|${ledgerLen}|${lastLedger}`;
}

async function pollNarrativeBackground(sessionId) {
  try {
    const resp = await fetch(`/api/narrative/background/${sessionId}`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.url && data.url !== currentBackgroundUrl) {
      currentBackgroundUrl = data.url;
      applyNarrativeBackground(data.url);
      // Background landed — clear the progress badge so it doesn't
      // linger past the crossfade.
      _setNarrativeBgProgress(null);
    }
  } catch { /* silently fail */ }
  // Poll the image generation queue scoped to this session for
  // AUTO-INITIATED background jobs only — explicit user clicks
  // (Illustrate moment, Generate scene image) get their own in-
  // message loader. Without the category filter, both would light
  // up for the same job. The endpoint returns active=false when
  // the current job belongs to a different category.
  try {
    const r = await fetch(
      `/api/image/generation-status?session_id=${encodeURIComponent(sessionId)}&category=auto_bg`,
    );
    if (!r.ok) return;
    const status = await r.json();
    if (status.active) {
      _setNarrativeBgProgress(status);
    } else {
      _setNarrativeBgProgress(null);
    }
  } catch { /* network blip — keep trying */ }
}

let _narrativeBgBadgeEl = null;
let _narrativeBgFadeTimer = null;

function _ensureNarrativeBgBadge() {
  if (_narrativeBgBadgeEl) return _narrativeBgBadgeEl;
  const mainArea = document.querySelector('.main-area');
  if (!mainArea) return null;
  const el = document.createElement('div');
  el.id = 'narrative-bg-progress';
  el.className = 'narrative-bg-progress';
  el.innerHTML = `
    <span class="narrative-bg-progress-label">Generating background…</span>
    <div class="narrative-bg-progress-bar"><div class="narrative-bg-progress-fill"></div></div>
    <span class="narrative-bg-progress-detail"></span>
  `;
  mainArea.appendChild(el);
  _narrativeBgBadgeEl = el;
  return el;
}

/** Render or hide the auto-bg badge based on a /generation-status payload.
 *  Pass null to hide. Hiding fades out so the user's eye can track that
 *  "the image just arrived" without the badge popping. */
function _setNarrativeBgProgress(status) {
  if (!status) {
    if (_narrativeBgBadgeEl) {
      _narrativeBgBadgeEl.classList.remove('visible');
      // Remove from DOM after the fade so subsequent shows recreate cleanly.
      if (_narrativeBgFadeTimer) clearTimeout(_narrativeBgFadeTimer);
      _narrativeBgFadeTimer = setTimeout(() => {
        if (_narrativeBgBadgeEl && _narrativeBgBadgeEl.parentNode) {
          _narrativeBgBadgeEl.parentNode.removeChild(_narrativeBgBadgeEl);
        }
        _narrativeBgBadgeEl = null;
      }, 400);
    }
    return;
  }
  const el = _ensureNarrativeBgBadge();
  if (!el) return;
  if (_narrativeBgFadeTimer) {
    clearTimeout(_narrativeBgFadeTimer);
    _narrativeBgFadeTimer = null;
  }
  el.classList.add('visible');
  const labelEl = el.querySelector('.narrative-bg-progress-label');
  const fillEl = el.querySelector('.narrative-bg-progress-fill');
  const barEl = el.querySelector('.narrative-bg-progress-bar');
  const detailEl = el.querySelector('.narrative-bg-progress-detail');

  const stage = status?.pre_queue?.stage || status?.stage || 'Generating background';
  labelEl.textContent = `${stage}…`;

  if (status.steps_total > 0) {
    const pct = Math.max(0, Math.min(100, (status.steps_done / status.steps_total) * 100));
    barEl.dataset.determinate = 'true';
    fillEl.style.width = `${pct}%`;
    detailEl.textContent = `step ${status.steps_done}/${status.steps_total} · ${Math.round(status.elapsed_s || 0)}s`;
  } else {
    barEl.dataset.determinate = 'false';
    fillEl.style.width = '';
    detailEl.textContent = status.elapsed_s ? `${Math.round(status.elapsed_s)}s` : '';
  }
}

function applyNarrativeBackground(url) {
  const mainArea = document.querySelector('.main-area');
  if (!mainArea) return;

  // Mark app as having an active background for text contrast CSS
  const appEl = document.getElementById('app');
  if (appEl && !appEl.dataset.bgActive) appEl.dataset.bgActive = 'true';

  // Create or update the background overlay element
  let bgEl = mainArea.querySelector('.narrative-bg-image');
  if (!bgEl) {
    bgEl = document.createElement('div');
    bgEl.className = 'narrative-bg-image';
    mainArea.insertBefore(bgEl, mainArea.firstChild);
  }

  // Crossfade: create a new layer, fade in, remove old
  const newBg = document.createElement('div');
  newBg.className = 'narrative-bg-image narrative-bg-fade-in';
  // Sanitize URL to prevent CSS injection (strip parens, quotes, semicolons)
  const safeUrl = url.replace(/[()'";\\\n\r]/g, '');
  newBg.style.backgroundImage = `url(${safeUrl})`;
  mainArea.insertBefore(newBg, mainArea.firstChild);

  // Trigger reflow then add the visible class
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      newBg.classList.add('narrative-bg-visible');
    });
  });

  // Remove old background after transition
  if (bgEl !== newBg) {
    setTimeout(() => bgEl.remove(), 1200);
  }
}

export function clearNarrativeBackground() {
  currentBackgroundUrl = null;
  const mainArea = document.querySelector('.main-area');
  if (!mainArea) return;
  const bgs = mainArea.querySelectorAll('.narrative-bg-image');
  bgs.forEach(bg => bg.remove());
  // Remove active background markers
  const appEl = document.getElementById('app');
  if (appEl) {
    delete appEl.dataset.bgActive;
    delete appEl.dataset.bgFrosted;
  }
}

// Collapse state persisted per section across re-renders
const _collapsedSections = new Set();

function _sectionHeader(id, icon, label, badge, { warn = false, defaultCollapsed = false } = {}) {
  // First render: apply default collapsed state
  if (defaultCollapsed && !_collapsedSections._initialized?.has(id)) {
    _collapsedSections.add(id);
  }
  if (!_collapsedSections._initialized) _collapsedSections._initialized = new Set();
  _collapsedSections._initialized.add(id);

  const collapsed = _collapsedSections.has(id);
  const chevron = `<svg class="ns-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><polyline points="6 9 12 15 18 9"/></svg>`;
  return `<div class="ns-section-header${warn ? ' ns-header-warn' : ''}" data-section-id="${id}" onclick="this.closest('.ns-section').classList.toggle('ns-collapsed');window._nsToggleCollapse('${id}')">
    ${icon}
    <span>${label}</span>
    ${badge ? `<span class="ns-badge">${badge}</span>` : ''}
    ${chevron}
  </div>`;
}

// Called from onclick — track collapse state across re-renders
window._nsToggleCollapse = function(id) {
  if (_collapsedSections.has(id)) _collapsedSections.delete(id);
  else _collapsedSections.add(id);
};

// Memory summary editing
let _requestLogList = [];       // full log array (oldest first)
let _requestLogIndex = 0;       // viewing offset from newest (0 = most recent)
let _requestLogTotal = 0;
let _requestLogLimit = 10;
let _requestLogVisible = false;
let _promptEditorVisible = false;
let _ltmPromptDefault = '';
let _ltmPromptCustom = '';
let _ltmPromptLoaded = false;
let _ltmSettingsLoaded = false;
let _ltmInterval = 10;
let _ltmModel = '';
let _ltmMode = 'standard';
let _ltmLedgerCeiling = 60;
let _ltmCompactionEnabled = true;
let _advancedVisible = false;
let _llmExtraction = true;
let _extractionInterval = 3;
let _extractionModel = '';
let _sceneContextRounds = 2;
let _autoBackground = false;
let _autoBgInterval = 4;
let _autoBgDistillerModel = '';
let _autoBgImageModel = '';
let _stateEnabled = true;
let _ledgerEnabled = true;
let _smartRetrieval = true;
let _smartRetrievalCount = 5;
let _continuousArchive = true;
let _archiveMinMessages = 75;
let _contextBudgetTokens = 1000;  // 0 = unlimited, otherwise max tokens for injected narrative context
let _recallToolsEnabled = false;
let _lorebookToolsEnabled = false;
let _toolsSectionVisible = false;
let _expandedArchiveIds = new Set();

window._nsToggleRequestLog = async function() {
  _requestLogVisible = !_requestLogVisible;
  if (_requestLogVisible) {
    const sessionId = app.state.currentSessionId;
    if (!sessionId) return;
    try {
      const resp = await fetch(`/api/ui/session/${sessionId}/request-log`);
      if (resp.ok) {
        const data = await resp.json();
        _requestLogList = data.logs || [];
        _requestLogTotal = data.total || 0;
        _requestLogIndex = 0;  // start at most recent
      }
    } catch { /* ignore */ }
  }
  renderNarrativeStateInspector();
};

window._nsRequestLogPrev = function() {
  if (_requestLogIndex < _requestLogTotal - 1) {
    _requestLogIndex++;
    renderNarrativeStateInspector();
  }
};

window._nsRequestLogNext = function() {
  if (_requestLogIndex > 0) {
    _requestLogIndex--;
    renderNarrativeStateInspector();
  }
};

window._nsChangeLogLimit = async function(val) {
  const v = parseInt(val, 10);
  if (isNaN(v) || v < 5 || v > 50) return;
  _requestLogLimit = v;
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ narrative_request_log_limit: v }),
    });
  } catch { /* ignore */ }
};


window._nsSaveState = async function() {
  const sessionId = app.state.currentSessionId;
  if (!sessionId) return;
  // Gather all state field inputs
  const inputs = document.querySelectorAll('.ns-state-field-input');
  const snapshot = {};
  inputs.forEach(inp => { snapshot[inp.dataset.field] = inp.value.trim(); });
  try {
    const resp = await fetch(`/api/ui/session/${sessionId}/state`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state_snapshot: snapshot }),
    });
    if (resp.ok) {
      if (narrativeState) narrativeState.state_snapshot = snapshot;
      showToast('State saved', 'success');
      renderNarrativeStateInspector();
    }
  } catch { /* ignore */ }
};

window._nsSaveLedger = async function() {
  const sessionId = app.state.currentSessionId;
  if (!sessionId || !narrativeState) return;
  try {
    const resp = await fetch(`/api/ui/session/${sessionId}/state`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ memory_ledger: narrativeState.memory_ledger || [] }),
    });
    if (resp.ok) showToast('Ledger saved', 'success');
  } catch { /* ignore */ }
};

window._nsToggleArchiveEntry = function(id) {
  if (_expandedArchiveIds.has(id)) {
    _expandedArchiveIds.delete(id);
  } else {
    _expandedArchiveIds.add(id);
  }
  renderNarrativeStateInspector();
};

window._nsDeleteArchiveEntry = async function(exchangeId) {
  try {
    const resp = await fetch(`/api/ui/archive/${encodeURIComponent(exchangeId)}`, { method: 'DELETE' });
    if (resp.ok) {
      showToast('Archive entry removed', 'success');
      _expandedArchiveIds.delete(exchangeId);
      if (narrativeState && narrativeState._archive) {
        narrativeState._archive = narrativeState._archive.filter(e => e.id !== exchangeId);
      }
      renderNarrativeStateInspector();
    } else {
      showToast('Failed to remove entry', 'error');
    }
  } catch { showToast('Failed to remove entry', 'error'); }
};

window._nsDeleteLedgerEntry = async function(index) {
  if (!narrativeState || !narrativeState.memory_ledger) return;
  narrativeState.memory_ledger.splice(index, 1);
  await window._nsSaveLedger();
  renderNarrativeStateInspector();
};

window._nsAddLedgerEntry = async function() {
  const catSelect = document.getElementById('ns-ledger-add-category');
  const contentInput = document.getElementById('ns-ledger-add-content');
  const roundInput = document.getElementById('ns-ledger-add-round');
  if (!catSelect || !contentInput || !narrativeState) return;
  const content = contentInput.value.trim();
  if (!content) return;
  const entry = {
    round_num: parseInt(roundInput?.value || '0', 10) || (narrativeState.message_count || 0),
    category: catSelect.value,
    content: content,
  };
  if (!narrativeState.memory_ledger) narrativeState.memory_ledger = [];
  narrativeState.memory_ledger.push(entry);
  await window._nsSaveLedger();
  renderNarrativeStateInspector();
};

window._nsChangeLtmLedgerCeiling = async function(val) {
  const v = parseInt(val, 10);
  if (isNaN(v) || (v !== 0 && v < 10) || v > 500) return;
  _ltmLedgerCeiling = v;
  try {
    const sessionId = chat.getActiveSessionId();
    if (!sessionId) return;
    await fetch(`/api/narrative/session/${sessionId}/memory-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ memory_ledger_ceiling: v }),
    });
  } catch { /* ignore */ }
};

window._nsChangeCompactionEnabled = async function(enabled) {
  _ltmCompactionEnabled = enabled;
  try {
    const sessionId = chat.getActiveSessionId();
    if (!sessionId) return;
    await fetch(`/api/narrative/session/${sessionId}/memory-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ memory_compaction_enabled: enabled }),
    });
  } catch { /* ignore */ }
  renderNarrativeStateInspector();
};

// LTM settings — load from global config (infrastructure settings) + session endpoint (per-session settings)
async function _loadLtmSettings() {
  if (_ltmSettingsLoaded) return;

  // Fetch global config once (for global-only settings + fallback)
  let globalData = null;
  try {
    globalData = await getToolSettings();
  } catch { /* ignore */ }

  // Load global-only infrastructure settings (always from global config)
  if (globalData) {
    _ltmModel = globalData.narrative_memory_model ?? '';
    _llmExtraction = globalData.narrative_llm_extraction ?? true;
    _extractionInterval = globalData.narrative_extraction_interval ?? 3;
    _extractionModel = globalData.narrative_extraction_model ?? '';
    _sceneContextRounds = globalData.narrative_scene_context_rounds ?? 2;
    _autoBackground = globalData.narrative_auto_background ?? false;
    _autoBgInterval = globalData.narrative_auto_background_interval ?? 4;
    _autoBgDistillerModel = globalData.narrative_auto_bg_distiller_model ?? '';
    _autoBgImageModel = globalData.narrative_auto_bg_image_model ?? '';
    _requestLogLimit = globalData.narrative_request_log_limit ?? 10;
    _archiveMinMessages = globalData.narrative_archive_min_messages ?? 75;
    _recallToolsEnabled = globalData.narrative_recall_tools_enabled ?? false;
    _lorebookToolsEnabled = globalData.narrative_lorebook_native_tools_enabled ?? true;
    // Context budget: backend config is in tokens. 0 = unlimited (default).
    // Historical note: UI used to show ~chars/4 tokens; now a 1:1 passthrough.
    const _rawBudget = globalData.narrative_context_budget ?? 0;
    _contextBudgetTokens = _rawBudget > 0 ? _rawBudget : 0;
  }

  // Try session-specific settings (returns effective = resolved values)
  let sessionLoaded = false;
  const sessionId = chat.getActiveSessionId();
  if (sessionId) {
    try {
      const resp = await fetch(`/api/narrative/session/${sessionId}/memory-settings`);
      if (resp.ok) {
        const data = await resp.json();
        const eff = data.effective || {};
        _ltmMode = eff.memory_enabled === false ? 'disabled' : (eff.memory_mode || 'standard');
        _stateEnabled = eff.memory_state_enabled ?? true;
        _ledgerEnabled = eff.memory_ledger_enabled ?? true;
        _continuousArchive = eff.memory_continuous_archive ?? true;
        _smartRetrieval = eff.smart_retrieval ?? true;
        _smartRetrievalCount = eff.smart_retrieval_count ?? 5;
        _ltmLedgerCeiling = eff.memory_ledger_ceiling ?? 60;
        _ltmCompactionEnabled = eff.memory_compaction_enabled ?? true;
        _ltmInterval = eff.memory_interval ?? 10;
        sessionLoaded = true;
      }
    } catch { /* fall through to global */ }
  }

  // Fall back to global config for per-session settings if no session data
  if (!sessionLoaded && globalData) {
    const enabled = globalData.narrative_memory_enabled ?? true;
    _ltmMode = enabled ? (globalData.narrative_memory_mode ?? 'standard') : 'disabled';
    _stateEnabled = globalData.narrative_memory_state_enabled ?? true;
    _ledgerEnabled = globalData.narrative_memory_ledger_enabled ?? true;
    _continuousArchive = globalData.narrative_memory_continuous_archive ?? true;
    _smartRetrieval = globalData.narrative_smart_retrieval ?? true;
    _smartRetrievalCount = globalData.narrative_smart_retrieval_count ?? 5;
    _ltmLedgerCeiling = globalData.narrative_memory_ledger_ceiling ?? 60;
    _ltmCompactionEnabled = globalData.narrative_memory_compaction_enabled ?? true;
    _ltmInterval = globalData.narrative_memory_interval ?? 10;
  }

  _ltmSettingsLoaded = true;
}

window._nsChangeLtmInterval = async function(val) {
  const v = parseInt(val, 10);
  if (isNaN(v) || v < 5 || v > 20) return;
  _ltmInterval = v;
  try {
    const sessionId = chat.getActiveSessionId();
    if (!sessionId) return;
    await fetch(`/api/narrative/session/${sessionId}/memory-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ memory_interval: v }),
    });
  } catch { /* ignore */ }
};

window._nsChangeLtmModel = async function(val) {
  _ltmModel = val;
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ narrative_memory_model: val }),
    });
  } catch { /* ignore */ }
};

window._nsChangeLtmMode = async function(val) {
  if (val !== 'lite' && val !== 'standard' && val !== 'disabled') return;
  _ltmMode = val;
  _ltmPromptLoaded = false; // Default prompt changes per mode
  const isDisabled = val === 'disabled';
  const payload = isDisabled
    ? { memory_enabled: false }
    : { memory_enabled: true, memory_mode: val };
  try {
    const sessionId = chat.getActiveSessionId();
    if (!sessionId) return;
    await fetch(`/api/narrative/session/${sessionId}/memory-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    showToast(`LTM: ${isDisabled ? 'disabled' : val}`, 'success');
  } catch { /* ignore */ }
  renderNarrativeStateInspector();
};

window._nsToggleAdvanced = function() {
  _advancedVisible = !_advancedVisible;
  renderNarrativeStateInspector();
};

window._nsToggleToolsSection = function() {
  _toolsSectionVisible = !_toolsSectionVisible;
  renderNarrativeStateInspector();
};

// ── Branches lifecycle (list / status / delete) ─────────────────────
// Hits /api/narrative/session/{id}/branches (GET) and the per-branch
// PATCH/DELETE endpoints. Storage widget reads /storage. All scoped to
// the active narrative session — no-op if there's no active session.

window._nsLoadBranches = async function() {
  const session = _activeNarrativeSession();
  if (!session) return;
  const listEl = document.getElementById('ns-branches-list');
  const storageEl = document.getElementById('ns-branches-storage');
  if (!listEl) return;
  listEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px">Loading…</div>';
  if (storageEl) storageEl.textContent = '';

  // Parallel — both endpoints are cheap reads.
  const [bResp, sResp] = await Promise.all([
    fetch(`/api/narrative/session/${encodeURIComponent(session.id)}/branches?include_stale=true`,
      { credentials: 'same-origin' }).catch(() => null),
    fetch(`/api/narrative/session/${encodeURIComponent(session.id)}/storage`,
      { credentials: 'same-origin' }).catch(() => null),
  ]);

  if (!bResp || !bResp.ok) {
    listEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px">No branch data.</div>';
    return;
  }
  const data = await bResp.json();
  const branches = data.branches || [];
  if (branches.length === 0) {
    listEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px">No branches recorded.</div>';
  } else {
    listEl.innerHTML = branches.map(b => {
      const status = b.status || 'unknown';
      const isMain = b.branch_id === 'main';
      const pill = status === 'active' ? '🟢' : status === 'archived' ? '📦' : status === 'stale' ? '⚪' : '·';
      return `
        <div class="ns-branch-row" data-branch-id="${escapeHtml(b.branch_id)}"
             style="display:flex;align-items:center;gap:var(--space-sm);padding:var(--space-xs) 0;border-bottom:1px solid var(--border-subtle)">
          <span>${pill}</span>
          <div style="flex:1;min-width:0">
            <div style="font-weight:600">${escapeHtml(b.branch_id)}${isMain ? ' <em style="color:var(--text-muted)">(main)</em>' : ''}</div>
            <div style="font-size:11px;color:var(--text-muted)">status: ${escapeHtml(status)} · created ${escapeHtml(b.created_at || '?')}</div>
          </div>
          ${!isMain ? `
            <button class="btn btn-sm" data-branch-action="archive" data-id="${escapeHtml(b.branch_id)}"
                    ${status === 'archived' ? 'disabled' : ''}>Archive</button>
            <button class="btn btn-sm" data-branch-action="activate" data-id="${escapeHtml(b.branch_id)}"
                    ${status === 'active' ? 'disabled' : ''}>Activate</button>
            <button class="btn btn-sm" data-branch-action="delete" data-id="${escapeHtml(b.branch_id)}">Delete</button>
          ` : '<em style="color:var(--text-muted);font-size:11px">undeletable</em>'}
        </div>`;
    }).join('');

    // Wire actions
    listEl.querySelectorAll('[data-branch-action]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const branchId = btn.dataset.id;
        const action = btn.dataset.branchAction;
        if (!branchId || !action) return;
        btn.disabled = true;
        try {
          if (action === 'archive' || action === 'activate') {
            const status = action === 'archive' ? 'archived' : 'active';
            const r = await fetch(
              `/api/narrative/session/${encodeURIComponent(session.id)}/branches/${encodeURIComponent(branchId)}/status`,
              {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                body: JSON.stringify({ status }),
              },
            );
            if (!r.ok) throw new Error(`status ${r.status}`);
          } else if (action === 'delete') {
            if (!confirm(`Delete branch "${branchId}" and all its content?`)) {
              btn.disabled = false;
              return;
            }
            const r = await fetch(
              `/api/narrative/session/${encodeURIComponent(session.id)}/branches/${encodeURIComponent(branchId)}?cascade=true`,
              { method: 'DELETE', credentials: 'same-origin' },
            );
            if (!r.ok) throw new Error(`status ${r.status}`);
          }
          await window._nsLoadBranches();
        } catch (err) {
          showToast?.(`Action failed: ${err.message || err}`, 'error');
          btn.disabled = false;
        }
      });
    });
  }

  // Storage rollup
  if (storageEl && sResp && sResp.ok) {
    const s = await sResp.json();
    const mb = (s.total_approx_bytes || 0) / (1024 * 1024);
    storageEl.textContent =
      `${s.total_branches || 0} branches · ${(s.total_ledger_entries || 0).toLocaleString()} ledger entries · ` +
      `${(s.total_archive_rows || 0).toLocaleString()} archive rows · ~${mb.toFixed(1)} MB`;
  }
};

window._nsToggleLayer = async function(key, checked) {
  if (key === 'narrative_memory_state_enabled') _stateEnabled = checked;
  else if (key === 'narrative_memory_ledger_enabled') _ledgerEnabled = checked;
  // Map long names to short session-endpoint names
  const _KEY_MAP = {
    'narrative_memory_state_enabled': 'memory_state_enabled',
    'narrative_memory_ledger_enabled': 'memory_ledger_enabled',
  };
  const shortKey = _KEY_MAP[key] || key;
  try {
    const sessionId = chat.getActiveSessionId();
    if (!sessionId) return;
    await fetch(`/api/narrative/session/${sessionId}/memory-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [shortKey]: checked }),
    });
    showToast(checked ? 'Layer enabled' : 'Layer disabled');
  } catch { showToast('Failed to update setting', 'error'); }
  renderNarrativeStateInspector();
};

window._nsChangeContextBudget = async function(val) {
  const tokens = parseInt(val, 10);
  if (isNaN(tokens) || tokens < 0) return;
  _contextBudgetTokens = tokens;
  // Backend takes tokens directly (1:1). 0 = unlimited.
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ narrative_context_budget: tokens }),
    });
  } catch { /* ignore */ }
  renderNarrativeStateInspector();
};

window._nsChangeArchiveMinMessages = async function(val) {
  const v = parseInt(val, 10);
  if (isNaN(v) || v < 0 || v > 500) return;
  _archiveMinMessages = v;
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ narrative_archive_min_messages: v }),
    });
  } catch { /* ignore */ }
};

window._nsToggleArchiveLayer = async function(checked) {
  _continuousArchive = checked;
  _smartRetrieval = checked;
  try {
    const sessionId = chat.getActiveSessionId();
    if (!sessionId) return;
    await fetch(`/api/narrative/session/${sessionId}/memory-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        memory_continuous_archive: checked,
        smart_retrieval: checked,
      }),
    });
    showToast(checked ? 'Archive enabled' : 'Archive disabled');
  } catch { showToast('Failed to update setting', 'error'); }
  renderNarrativeStateInspector();
};

window._nsChangeAdvanced = async function(key, value) {
  const payload = {};
  const _BOOL_KEYS = new Set([
    'narrative_llm_extraction', 'narrative_auto_background',
    'narrative_smart_retrieval', 'narrative_memory_continuous_archive',
    'narrative_recall_tools_enabled', 'narrative_lorebook_native_tools_enabled',
  ]);
  const _INT_KEYS = new Set([
    'narrative_extraction_interval', 'narrative_scene_context_rounds',
    'narrative_auto_background_interval', 'narrative_smart_retrieval_count',
  ]);
  // Per-session keys that must be written to the session endpoint (not global config)
  const _SESSION_KEY_MAP = {
    'narrative_smart_retrieval': 'smart_retrieval',
    'narrative_memory_continuous_archive': 'memory_continuous_archive',
    'narrative_smart_retrieval_count': 'smart_retrieval_count',
  };

  if (_BOOL_KEYS.has(key)) {
    const boolVal = value === true || value === 'true';
    payload[key] = boolVal;
    if (key === 'narrative_llm_extraction') _llmExtraction = boolVal;
    else if (key === 'narrative_auto_background') _autoBackground = boolVal;
    else if (key === 'narrative_smart_retrieval') _smartRetrieval = boolVal;
    else if (key === 'narrative_memory_continuous_archive') _continuousArchive = boolVal;
    else if (key === 'narrative_recall_tools_enabled') _recallToolsEnabled = boolVal;
    else if (key === 'narrative_lorebook_native_tools_enabled') _lorebookToolsEnabled = boolVal;
  } else if (_INT_KEYS.has(key)) {
    const intVal = parseInt(value, 10);
    if (isNaN(intVal)) return;
    payload[key] = intVal;
    if (key === 'narrative_extraction_interval') _extractionInterval = intVal;
    else if (key === 'narrative_scene_context_rounds') _sceneContextRounds = intVal;
    else if (key === 'narrative_auto_background_interval') _autoBgInterval = intVal;
    else if (key === 'narrative_smart_retrieval_count') _smartRetrievalCount = intVal;
  } else {
    payload[key] = value;
    if (key === 'narrative_extraction_model') _extractionModel = value;
    else if (key === 'narrative_auto_bg_distiller_model') _autoBgDistillerModel = value;
    else if (key === 'narrative_auto_bg_image_model') _autoBgImageModel = value;
  }
  try {
    if (key in _SESSION_KEY_MAP) {
      const sessionId = chat.getActiveSessionId();
      if (!sessionId) return;
      const shortKey = _SESSION_KEY_MAP[key];
      await fetch(`/api/narrative/session/${sessionId}/memory-settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [shortKey]: payload[key] }),
      });
    } else {
      await fetch('/api/config/tools', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    }
  } catch { /* ignore */ }
  renderNarrativeStateInspector();
};

window._nsTogglePromptEditor = async function() {
  _promptEditorVisible = !_promptEditorVisible;
  if (_promptEditorVisible && !_ltmPromptLoaded) {
    const sessionId = app.state.currentSessionId;
    if (sessionId) {
      try {
        const resp = await fetch(`/api/ui/session/${sessionId}/ltm-prompt`);
        if (resp.ok) {
          const data = await resp.json();
          _ltmPromptDefault = data.default_prompt || '';
          _ltmPromptCustom = data.custom_prompt || '';
          _ltmPromptLoaded = true;
        }
      } catch { /* ignore */ }
    }
  }
  renderNarrativeStateInspector();
};

window._nsSavePrompt = async function() {
  const textarea = document.getElementById('ns-prompt-textarea');
  if (!textarea) return;
  const val = textarea.value.trim();
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ narrative_memory_prompt: val }),
    });
    _ltmPromptCustom = val;
    _promptEditorVisible = false;
    _ltmPromptLoaded = false; // Reload on next open
    showToast('LTM prompt saved', 'success');
    renderNarrativeStateInspector();
  } catch { /* ignore */ }
};

window._nsResetPrompt = function() {
  const textarea = document.getElementById('ns-prompt-textarea');
  if (textarea) textarea.value = _ltmPromptDefault;
};

function _estimateTokens(text) {
  return Math.ceil((text || '').length / 4);
}

// Color palette for context blocks (NovelAI-inspired)
const _BLOCK_COLORS = {
  character_card: '#7c6cd9',
  example_dialogue: '#8b78e6',
  authors_note: '#d97c4c',
  narrative_memory: '#4caf50',
  previous_events: '#2196f3',
  known_relationships: '#e91e63',
  character_relationships: '#e91e63',
  character_states: '#9c27b0',
  current_scene: '#00bcd4',
  active_plots: '#ff9800',
  established_facts: '#607d8b',
  consistency_warnings: '#f44336',
  'preset:system_prompt': '#795548',
  state_snapshot: '#00897b',
  story_memory: '#4caf50',
  'preset:jailbreak': '#f44336',
  'preset:post_history': '#ff5722',
  'preset:author_note': '#d97c4c',
};

function _blockColor(label) {
  if (_BLOCK_COLORS[label]) return _BLOCK_COLORS[label];
  if (label.startsWith('lore:')) return '#cddc39';
  if (label.startsWith('preset:')) return '#795548';
  return '#9e9e9e';
}

function _blockLabel(label) {
  const labels = {
    character_card: 'Character Card',
    example_dialogue: 'Example Dialogue',
    authors_note: "Author's Note",
    previous_events: 'Chat Summaries',
    character_relationships: 'Relationships',
    character_states: 'Characters',
    current_scene: 'Scene',
    active_plots: 'Plots',
    established_facts: 'Facts',
    consistency_warnings: 'Consistency',
    state_snapshot: 'Current State',
    story_memory: 'Story Memory',
    'preset:system_prompt': 'System Prompt',
    'preset:jailbreak': 'Jailbreak',
    'preset:post_history': 'Post History',
    'preset:author_note': "Author's Note",
  };
  if (labels[label]) return labels[label];
  if (label.startsWith('lore:')) return 'Lore: ' + label.slice(5);
  if (label.startsWith('preset:')) return 'Preset: ' + label.slice(7);
  return label;
}

function _renderRequestLog(log) {
  if (!log) return '<div class="ns-empty-hint"><span>No request log yet. Send a message first.</span></div>';

  const blocks = log.context_blocks || [];
  const included = blocks.filter(b => b.included);
  const excluded = blocks.filter(b => !b.included);
  const totalContextTokens = log.context_tokens_total || 0;
  const budget = log.context_budget || 0;
  const msgTokens = log.total_token_estimate || 0;

  let html = '';

  // Summary bar
  html += `<div class="ns-rlog-summary">
    <span>${log.total_messages || 0} messages</span>
    <span class="ns-rlog-sep">&middot;</span>
    <span>~${msgTokens.toLocaleString()} tokens total</span>
    <span class="ns-rlog-sep">&middot;</span>
    <span>Context: ${totalContextTokens}/${budget}</span>
  </div>`;

  // Segmented bar (proportional, NovelAI-style)
  if (included.length > 0) {
    const totalTokens = included.reduce((s, b) => s + b.token_estimate, 0) || 1;
    html += '<div class="ns-rlog-bar">';
    for (const block of included) {
      const pct = Math.max(2, (block.token_estimate / totalTokens) * 100);
      const color = _blockColor(block.label);
      const lbl = _blockLabel(block.label);
      html += `<div class="ns-rlog-bar-seg" style="width:${pct.toFixed(1)}%;background:${color}" title="${lbl}: ~${block.token_estimate} tokens"></div>`;
    }
    html += '</div>';
  }

  // Message breakdown
  const mc = log.message_counts || {};
  html += `<div class="ns-rlog-msg-counts">
    <span class="ns-rlog-role">sys: ${mc.system || 0}</span>
    <span class="ns-rlog-role">user: ${mc.user || 0}</span>
    <span class="ns-rlog-role">asst: ${mc.assistant || 0}</span>
    ${log.model ? `<span class="ns-rlog-model">${escapeHtml(log.model)}</span>` : ''}
  </div>`;

  // Block details
  html += '<div class="ns-rlog-blocks">';
  for (const block of included) {
    const color = _blockColor(block.label);
    const lbl = _blockLabel(block.label);
    const content = block.content || '';
    const isLong = content.length > 200;
    html += `<div class="ns-rlog-block${isLong ? ' ns-rlog-expandable' : ''}"${isLong ? ' onclick="this.classList.toggle(\'ns-rlog-expanded\')"' : ''}>
      <div class="ns-rlog-block-header">
        <span class="ns-rlog-block-dot" style="background:${color}"></span>
        <span class="ns-rlog-block-label">${lbl}</span>
        <span class="ns-rlog-block-tokens">~${block.token_estimate} tok${isLong ? ' &middot; click to expand' : ''}</span>
      </div>
      <div class="ns-rlog-block-preview">${escapeHtml(content)}</div>
    </div>`;
  }

  // Excluded blocks
  if (excluded.length > 0) {
    html += '<div class="ns-rlog-excluded-header">Excluded (over budget)</div>';
    for (const block of excluded) {
      const lbl = _blockLabel(block.label);
      html += `<div class="ns-rlog-block ns-rlog-block-excluded">
        <div class="ns-rlog-block-header">
          <span class="ns-rlog-block-dot" style="background:#666"></span>
          <span class="ns-rlog-block-label">${lbl}</span>
          <span class="ns-rlog-block-tokens">~${block.token_estimate} tok</span>
        </div>
      </div>`;
    }
  }

  html += '</div>';
  return html;
}

function renderNarrativeStateInspector() {
  const container = document.getElementById('chat-settings-tab');
  if (!container) return;

  // Don't re-render while the user is editing the prompt
  if (_promptEditorVisible) return;

  // Load LTM settings on first render
  if (!_ltmSettingsLoaded) _loadLtmSettings().then(() => renderNarrativeStateInspector());

  const emptyEl = document.getElementById('chat-state-empty');
  if (emptyEl) emptyEl.style.display = 'none';

  // Use empty fallback so LTM settings always render even before first message
  const state = narrativeState || { message_count: 0, memory_summary: '' };
  let html = '<div class="narrative-state-panel">';

  // ── Message count (top bar) ──
  html += `<div class="ns-topbar">
    <span class="ns-msg-count">${state.message_count || 0} messages tracked</span>
    <button class="ns-rlog-btn${_requestLogVisible ? ' active' : ''}" onclick="window._nsToggleRequestLog()" title="View request log">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
      Logs
    </button>
  </div>`;

  // ── Request Log Panel (toggled) ──
  if (_requestLogVisible) {
    const _currentLog = _requestLogTotal > 0
      ? _requestLogList[_requestLogTotal - 1 - _requestLogIndex]
      : null;
    const _msgIdx = _currentLog?.message_index;

    let _navHtml = '<div class="ns-rlog-nav">';
    _navHtml += `<button class="ns-rlog-nav-btn" onclick="window._nsRequestLogPrev()" ${_requestLogIndex >= _requestLogTotal - 1 ? 'disabled' : ''} title="Older">&#9664;</button>`;
    _navHtml += `<span class="ns-rlog-nav-label">${_msgIdx != null ? `Msg ${_msgIdx}` : '\u2014'} (${_requestLogIndex + 1}/${_requestLogTotal || 0})</span>`;
    _navHtml += `<button class="ns-rlog-nav-btn" onclick="window._nsRequestLogNext()" ${_requestLogIndex <= 0 ? 'disabled' : ''} title="Newer">&#9654;</button>`;
    _navHtml += '</div>';

    html += `<div class="ns-rlog-panel">${_navHtml}${_renderRequestLog(_currentLog)}</div>`;
  }

  // ── LTM Settings (master controls — shown first) ──
  const _modeHints = {
    disabled: 'The AI has no memory of past events beyond what fits in the chat window',
    lite: 'Keeps a brief bullet-point timeline of key events \u2014 uses fewer tokens',
    standard: 'Keeps a detailed narrative record that grows with the conversation',
  };
  const _isDisabled = _ltmMode === 'disabled';
  html += `<div class="ns-ltm-settings">
    <div class="ns-ltm-row">
      <label class="ns-ltm-label ns-tip" data-tip="Gives the AI a long-term memory of your story. Without this, the AI forgets earlier events once they scroll out of the chat window." for="ns-ltm-mode">Memory</label>
      <select class="ns-ltm-select" id="ns-ltm-mode" onchange="window._nsChangeLtmMode(this.value)">
        <option value="disabled"${_ltmMode === 'disabled' ? ' selected' : ''}>Off</option>
        <option value="lite"${_ltmMode === 'lite' ? ' selected' : ''}>Lite</option>
        <option value="standard"${_ltmMode === 'standard' ? ' selected' : ''}>Standard</option>
      </select>
    </div>
    <div class="ns-ltm-hint-row"><span class="ns-ltm-hint">${_modeHints[_ltmMode] || _modeHints.standard}</span></div>
    <div class="ns-ltm-row"${_isDisabled || !_ledgerEnabled ? ' style="opacity:0.5;pointer-events:none"' : ''}>
      <label class="ns-ltm-label ns-tip" data-tip="How many memory entries to keep. When the limit is reached, older entries are merged together to save space. Use Unlimited for models with large context windows (32K+) so nothing is lost." for="ns-ltm-ledger-ceiling">Max entries</label>
      <select class="ns-ltm-select" id="ns-ltm-ledger-ceiling" onchange="window._nsChangeLtmLedgerCeiling(this.value)"${_isDisabled || !_ledgerEnabled ? ' disabled' : ''}>
        ${[20, 40, 60, 100, 200, 500].map(v =>
          `<option value="${v}"${v === _ltmLedgerCeiling ? ' selected' : ''}>${v}</option>`
        ).join('')}
        <option value="0"${_ltmLedgerCeiling === 0 ? ' selected' : ''}>Unlimited</option>
      </select>
    </div>
    <div class="ns-ltm-row"${_isDisabled || !_ledgerEnabled || _ltmLedgerCeiling === 0 ? ' style="opacity:0.5;pointer-events:none"' : ''}>
      <label class="ns-adv-toggle-label" style="flex:1">
        <input type="checkbox" ${_ltmCompactionEnabled ? 'checked' : ''}${_isDisabled || !_ledgerEnabled || _ltmLedgerCeiling === 0 ? ' disabled' : ''}
          onchange="window._nsChangeCompactionEnabled(this.checked)">
        <span class="ns-ltm-label ns-tip" data-tip="When the entry limit is reached, the AI combines older entries into shorter summaries so nothing is completely forgotten. Turn this off if your model has a very large context window and can hold the full timeline.">Merge old entries</span>
      </label>
    </div>
    <div class="ns-ltm-row"${_isDisabled ? ' style="opacity:0.5;pointer-events:none"' : ''}>
      <label class="ns-ltm-label ns-tip" data-tip="How many messages between memory updates. Lower values keep the memory more up-to-date, but each update uses one background AI call." for="ns-ltm-interval">Refresh every</label>
      <select class="ns-ltm-select" id="ns-ltm-interval" onchange="window._nsChangeLtmInterval(this.value)">
        ${[5,6,7,8,9,10,12,14,16,18,20].map(n =>
          `<option value="${n}"${n === _ltmInterval ? ' selected' : ''}>${n} msgs</option>`
        ).join('')}
      </select>
    </div>
    <div class="ns-ltm-row"${_isDisabled ? ' style="opacity:0.5;pointer-events:none"' : ''}>
      <label class="ns-ltm-label ns-tip" data-tip="Which AI model writes the memory summaries. Pick a fast/cheap model to save costs, or leave on Default to use whatever model you're chatting with." for="ns-ltm-model">Refresh model</label>
      <select class="ns-ltm-select ns-ltm-model-select" id="ns-ltm-model">
        <option value="">Default (chat model)</option>
      </select>
    </div>
    <div class="ns-ltm-row">
      <label class="ns-ltm-label ns-tip" data-tip="How many past messages to keep debug logs for. Each log shows exactly what the AI received, useful for troubleshooting.">Log history</label>
      <select class="ns-ltm-select" onchange="window._nsChangeLogLimit(this.value)">
        ${[5,10,15,20,30,50].map(v =>
          `<option value="${v}"${v === _requestLogLimit ? ' selected' : ''}>${v}</option>`
        ).join('')}
      </select>
    </div>
    <div class="ns-ltm-row"${_isDisabled ? ' style="opacity:0.5;pointer-events:none"' : ''}>
      <button class="ns-ltm-prompt-btn${_promptEditorVisible ? ' active' : ''}" onclick="window._nsTogglePromptEditor()"
        title="Customize the system prompt used when generating the state and memory">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
        Edit Refresh Prompt
      </button>
    </div>
  </div>`;

  // ── Prompt Editor (toggled) ──
  if (_promptEditorVisible) {
    const promptVal = _ltmPromptCustom || _ltmPromptDefault;
    html += `<div class="ns-prompt-editor">
      <div class="ns-prompt-editor-hint">
        Customize the system prompt sent to the LLM when it generates the rolling memory summary.
        The prompt is auto-detected based on card type (character, narrator, or ensemble).
        Use <code>${'\\{char_name\\}'}</code>, <code>${'\\{previous_context\\}'}</code>, and <code>${'\\{word_target\\}'}</code> as placeholders.
      </div>
      <textarea class="ns-prompt-textarea" id="ns-prompt-textarea" rows="10">${escapeHtml(promptVal)}</textarea>
      <div class="ns-prompt-editor-actions">
        <button class="btn btn-sm" onclick="window._nsResetPrompt()" title="Reset to the auto-detected default for this card type (character, narrator, or ensemble)">Reset Default</button>
        <span style="flex:1"></span>
        <button class="btn btn-sm" onclick="window._nsTogglePromptEditor()">Cancel</button>
        <button class="btn btn-sm btn-primary" onclick="window._nsSavePrompt()">Save</button>
      </div>
    </div>`;
  }

  // ── Branches & storage (read-only observability + lifecycle controls) ──
  // The narrative engine maintains a tree of branches per session (main +
  // any user-created forks). This block surfaces them with status pills,
  // storage rollup, and per-branch actions (archive / activate / delete).
  html += `<details class="ns-branches" style="margin-top:var(--space-sm)">
    <summary style="cursor:pointer;font-weight:600">Branches &amp; storage</summary>
    <div id="ns-branches-body" style="margin-top:var(--space-sm)">
      <button class="btn btn-sm" onclick="window._nsLoadBranches()">Refresh</button>
      <div id="ns-branches-list" style="margin-top:var(--space-sm)"></div>
      <div id="ns-branches-storage" style="margin-top:var(--space-sm);font-size:12px;color:var(--text-muted)"></div>
    </div>
  </details>`;

  // ── Archiving & Retrieval Settings ──
  html += `<div class="ns-advanced">
    <button class="ns-advanced-toggle" onclick="window._nsToggleAdvanced()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="10" height="10"
        style="transform:rotate(${_advancedVisible ? '90' : '0'}deg);transition:transform .15s">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
      Archiving &amp; Retrieval
    </button>`;

  if (_advancedVisible) {
    html += `<div class="ns-advanced-body">
      <div class="ns-adv-group">
        <label class="ns-adv-toggle-label">
          <input type="checkbox" ${_continuousArchive ? 'checked' : ''}
            onchange="window._nsChangeAdvanced('narrative_memory_continuous_archive', this.checked)">
          <span class="ns-tip" data-tip="Saves each conversation exchange so the AI can search through them later. Exchanges are summarized and indexed automatically in the background.">Record exchanges</span>
        </label>
        <div class="ns-adv-hint">Save conversations for the AI to search through later</div>
      </div>

      <div class="ns-adv-group">
        <label class="ns-adv-toggle-label">
          <input type="checkbox" ${_smartRetrieval ? 'checked' : ''}
            onchange="window._nsChangeAdvanced('narrative_smart_retrieval', this.checked)">
          <span class="ns-tip" data-tip="Before each reply, the AI searches its archive for past moments that relate to what's happening now and re-reads them. This is how it 'remembers' specific earlier scenes.">Recall past scenes</span>
        </label>
        <div class="ns-adv-hint">The AI re-reads relevant past moments before each reply</div>
        <div class="ns-adv-row"${!_smartRetrieval ? ' style="opacity:0.5;pointer-events:none"' : ''}>
          <label class="ns-ltm-label ns-tip" data-tip="How many past exchanges the AI can pull up per reply. More gives richer recall but uses more of the context window.">Recall up to</label>
          <select class="ns-ltm-select" onchange="window._nsChangeAdvanced('narrative_smart_retrieval_count', this.value)">
            ${[1,2,3,5,8,10,15,20].map(n =>
              `<option value="${n}"${n === _smartRetrievalCount ? ' selected' : ''}>${n} exchanges</option>`
            ).join('')}
          </select>
        </div>
      </div>

      <div class="ns-adv-group" style="border-top:1px solid var(--border-light);padding-top:6px;margin-top:4px">
        <label class="ns-adv-toggle-label">
          <span class="ns-tip" data-tip="How much memory the AI is allowed to include in each message. Unlimited means every memory entry is included \u2014 best if your model has a large context window (32K+). Lower values save room for conversation but the AI may forget older events.">Memory size limit</span>
        </label>
        <div class="ns-adv-row">
          <select class="ns-ltm-select" onchange="window._nsChangeContextBudget(this.value)">
            ${[0, 250, 500, 1000, 2000, 4000, 8000, 16000].map(n =>
              `<option value="${n}"${n === _contextBudgetTokens ? ' selected' : ''}>${n === 0 ? 'Unlimited' : n + ' tokens'}</option>`
            ).join('')}
          </select>
        </div>
        <div class="ns-adv-hint">${_contextBudgetTokens === 0 ? 'All memories included every message (best for large-context models)' : `The AI gets ~${_contextBudgetTokens} tokens of memory per message`}</div>
      </div>

    </div>`;
  }

  html += `</div>`;

  // ── LLM Tools (recall + lorebook tools) ──
  html += `<div class="ns-advanced" style="margin-top:var(--space-xs)">
    <button class="ns-advanced-toggle" onclick="window._nsToggleToolsSection()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="10" height="10"
        style="transform:rotate(${_toolsSectionVisible ? '90' : '0'}deg);transition:transform .15s">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
      LLM Tools
    </button>`;

  if (_toolsSectionVisible) {
    html += `<div class="ns-advanced-body">
      <div class="ns-adv-group">
        <label class="ns-adv-toggle-label">
          <input type="checkbox" ${_recallToolsEnabled ? 'checked' : ''}
            onchange="window._nsChangeAdvanced('narrative_recall_tools_enabled', this.checked)">
          <span class="ns-tip" data-tip="Let the AI look up characters, facts, plot threads, and past scenes from its memory during a turn. Instead of always pre-injecting everything, the AI asks for what it needs.">Memory recall tools</span>
        </label>
        <div class="ns-adv-hint">The AI can query its own memory mid-turn</div>
      </div>
      <div class="ns-adv-group">
        <label class="ns-adv-toggle-label">
          <input type="checkbox" ${_lorebookToolsEnabled ? 'checked' : ''}
            onchange="window._nsChangeAdvanced('narrative_lorebook_native_tools_enabled', this.checked)">
          <span class="ns-tip" data-tip="Let the AI search, create, update, and delete lorebook entries during a turn. The AI can check established lore for consistency and record new world details as the story progresses.">Lorebook tools</span>
        </label>
        <div class="ns-adv-hint">The AI can read and write world info as the story unfolds</div>
      </div>
    </div>`;
  }

  html += `</div>`;

  // ── Data sections (dimmed when memory is off) ──
  html += `<div class="ns-ltm-data-sections${_isDisabled ? ' ns-ltm-disabled' : ''}">`;
  if (_isDisabled) {
    html += `<div class="ns-ltm-disabled-banner">Memory is turned off &mdash; the AI will only remember what fits in the chat window</div>`;
  }

  // ── Current State (snapshot) ──
  const stateSnapshot = state.state_snapshot || {};
  const stateFields = Object.entries(stateSnapshot).filter(([, v]) => v);
  const stateTokens = _estimateTokens(stateFields.map(([k, v]) => `${k}: ${v}`).join('\n'));
  const _stateOff = !_isDisabled && !_stateEnabled;
  html += `<div class="ns-section ns-state${_stateOff ? ' ns-layer-off' : ''}">
    <div class="ns-section-header">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      ${!_isDisabled ? `<input type="checkbox" class="ns-layer-toggle"
        ${_stateEnabled ? 'checked' : ''}
        onclick="event.stopPropagation();window._nsToggleLayer('narrative_memory_state_enabled',this.checked)"
        title="Toggle auto-generation and injection of the state snapshot">` : ''}
      <span class="ns-tip" data-tip="A quick summary of what's happening right now \u2014 where the characters are, what they're doing, and the current mood. Updated every refresh.">Current State</span>
      ${!_stateEnabled && !_isDisabled ? '<span class="ns-layer-off-badge">(auto off)</span>' : ''}
      ${stateTokens > 0 ? `<span class="ns-token-badge">~${stateTokens} tok</span>` : ''}
    </div>
    <div class="ns-state-fields">
      ${stateFields.length > 0 ? stateFields.map(([key, value]) =>
        `<div class="ns-state-field">
          <span class="ns-state-label">${escapeHtml(key.replace(/_/g, ' '))}</span>
          <span class="ns-state-value">${escapeHtml(value)}</span>
        </div>`
      ).join('') : '<div class="ns-ledger-empty">The current state will appear after the first memory refresh</div>'}
    </div>
    <div class="ns-memory-editor hidden" id="ns-state-editor">
      ${(stateFields.length > 0 ? stateFields : [['', '']]).map(([key, value]) =>
        `<div class="ns-state-field">
          <label class="ns-state-label">${escapeHtml(key.replace(/_/g, ' '))}</label>
          <input class="ns-state-field-input" data-field="${escapeHtml(key)}" value="${escapeHtml(value)}" />
        </div>`
      ).join('')}
      <div class="ns-memory-editor-actions">
        <button class="btn btn-sm" onclick="document.getElementById('ns-state-editor').classList.add('hidden')">Cancel</button>
        <button class="btn btn-sm btn-primary" onclick="window._nsSaveState()">Save</button>
      </div>
    </div>
    <span class="ns-edit-btn" onclick="document.getElementById('ns-state-editor').classList.toggle('hidden')" title="Edit state fields">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
    </span>
  </div>`;

  // ── Memory Ledger ──
  const ledger = state.memory_ledger || [];
  const ledgerTokens = _estimateTokens(ledger.map(e => `[R${e.round_num}|${e.category}] ${e.content}`).join('\n'));
  const ledgerPct = _ltmLedgerCeiling > 0 ? Math.min(100, Math.round((ledger.length / _ltmLedgerCeiling) * 100)) : 0;
  const _badgeColorMap = {
    relationship_shift: '#2196f3', alliance: '#2196f3', npc_relationship: '#2196f3',
    discovery: '#4caf50', shared_discovery: '#4caf50', lore_reveal: '#4caf50',
    commitment: '#9c27b0', group_decision: '#9c27b0', party_decision: '#9c27b0',
    consequence: '#f44336', conflict: '#f44336',
    emotional_milestone: '#ff9800', character_development: '#ff9800',
    world_change: '#e65100',
    quest_update: '#009688', resource_change: '#009688', rule_established: '#009688',
  };
  const _ledgerOff = !_isDisabled && !_ledgerEnabled;
  html += `<div class="ns-section ns-ledger${_ledgerOff ? ' ns-layer-off' : ''}">
    <div class="ns-section-header">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z"/><line x1="9" y1="22" x2="15" y2="22"/></svg>
      ${!_isDisabled ? `<input type="checkbox" class="ns-layer-toggle"
        ${_ledgerEnabled ? 'checked' : ''}
        onclick="event.stopPropagation();window._nsToggleLayer('narrative_memory_ledger_enabled',this.checked)"
        title="Toggle auto-generation and injection of the memory ledger">` : ''}
      <span class="ns-tip" data-tip="A growing timeline of important story events \u2014 discoveries, conflicts, relationship changes, and plot turns. The AI reads this to remember what happened earlier in the conversation.">Memory Ledger</span>
      ${!_ledgerEnabled && !_isDisabled ? '<span class="ns-layer-off-badge">(auto off)</span>' : ''}
      ${ledgerTokens > 0 ? `<span class="ns-token-badge">~${ledgerTokens} tok</span>` : ''}
      <span class="ns-ledger-count" title="${ledger.length}/${_ltmLedgerCeiling > 0 ? _ltmLedgerCeiling : '∞'} entries">${ledger.length}/${_ltmLedgerCeiling > 0 ? _ltmLedgerCeiling : '∞'}</span>
    </div>
    <div class="ns-ledger-progress" title="${ledgerPct}% of ceiling">
      <div class="ns-ledger-progress-bar" style="width:${ledgerPct}%"></div>
    </div>
    <div class="ns-ledger-entries">
      ${ledger.length > 0 ? ledger.map((e, i) => {
        const color = _badgeColorMap[e.category] || '#9e9e9e';
        return `<div class="ns-ledger-entry">
          <span class="ns-ledger-badge" style="background:${color}20;color:${color};border-color:${color}40">[R${e.round_num}|${escapeHtml(e.category)}]</span>
          <span class="ns-ledger-content">${escapeHtml(e.content)}</span>
          <button class="ns-ledger-delete" onclick="window._nsDeleteLedgerEntry(${i})" title="Remove entry">&times;</button>
        </div>`;
      }).join('') : '<div class="ns-ledger-empty">No memories yet. Events will appear after the first summary refresh.</div>'}
    </div>
    <div class="ns-ledger-add">
      <select id="ns-ledger-add-category" class="ns-ltm-select">
        <option value="discovery">discovery</option>
        <option value="relationship_shift">relationship</option>
        <option value="world_change">world change</option>
        <option value="commitment">commitment</option>
        <option value="consequence">consequence</option>
        <option value="emotional_milestone">emotional</option>
      </select>
      <input id="ns-ledger-add-content" class="ns-ledger-add-input" placeholder="Add a memory entry..." />
      <input id="ns-ledger-add-round" class="ns-ledger-add-round" type="number" value="${state.message_count || 0}" title="Round number" />
      <button class="btn btn-sm" onclick="window._nsAddLedgerEntry()">+</button>
    </div>
  </div>`;


  // ── Embedded Archive ──
  const archiveExchanges = state._archive || [];
  const archiveTokens = _estimateTokens(archiveExchanges.map(e => e.summary || '').join('\n'));
  const _archiveOff = !_isDisabled && !_continuousArchive && !_smartRetrieval;
  html += `<div class="ns-section ns-archive${_archiveOff ? ' ns-layer-off' : ''}">
    <div class="ns-section-header">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M21 8v13H3V8"/><path d="M1 3h22v5H1z"/><path d="M10 12h4"/></svg>
      ${!_isDisabled ? `<input type="checkbox" class="ns-layer-toggle"
        ${_continuousArchive || _smartRetrieval ? 'checked' : ''}
        onclick="event.stopPropagation();window._nsToggleArchiveLayer(this.checked)"
        title="Toggle archive recording and retrieval">` : ''}
      <span class="ns-tip" data-tip="Past conversation exchanges saved for smart recall. When the current scene is similar to something that happened before, the AI can pull up those old exchanges to remember the details.">Embedded Archive</span>
      ${_archiveOff ? '<span class="ns-layer-off-badge">(auto off)</span>' : ''}
      ${archiveTokens > 0 ? `<span class="ns-token-badge">~${archiveTokens} tok</span>` : ''}
      <span class="ns-ledger-count">${archiveExchanges.length} exchanges</span>
    </div>
    <div class="ns-archive-settings" style="padding:4px 8px;display:flex;align-items:center;gap:6px;font-size:10px;color:var(--text-muted);border-bottom:1px solid var(--border-light)">
      <label style="white-space:nowrap" title="The AI won't search its archive until this many messages have been exchanged. Before this point the full conversation is still visible to the AI, so archive recall isn't needed yet. Set to 0 to always search.">Start recall after</label>
      <input type="number" min="0" max="500" step="5" value="${_archiveMinMessages}"
        style="width:52px;padding:2px 4px;font-size:10px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--input-bg);color:var(--text-primary)"
        onchange="window._nsChangeArchiveMinMessages(parseInt(this.value))"
        onclick="event.stopPropagation()">
      <span>messages</span>
      <span style="opacity:0.5">(0 = always)</span>
    </div>
    <div class="ns-archive-entries">`;

  if (archiveExchanges.length > 0) {
    for (const ex of archiveExchanges) {
      const isExpanded = _expandedArchiveIds.has(ex.id);
      const safeId = escapeHtml(ex.id);
      html += `<div class="ns-archive-entry${isExpanded ? ' ns-archive-expanded' : ''}" onclick="window._nsToggleArchiveEntry('${safeId}')" style="cursor:pointer">
        <span class="ns-archive-turn">R${ex.turn_number || '?'}</span>
        <span class="ns-archive-summary">${escapeHtml(ex.summary || '(no summary)')}</span>
        <span class="ns-archive-chevron">${isExpanded ? '▾' : '▸'}</span>
        <button class="ns-kg-delete" onclick="event.stopPropagation();window._nsDeleteArchiveEntry('${safeId}')" title="Remove">&times;</button>
      </div>`;
      if (isExpanded) {
        html += `<div class="ns-archive-exchange">
          <div class="ns-archive-speaker ns-archive-user"><span class="ns-archive-role">User</span><span class="ns-archive-text">${escapeHtml(ex.user_content || '')}</span></div>
          <div class="ns-archive-speaker ns-archive-asst"><span class="ns-archive-role">Asst</span><span class="ns-archive-text">${escapeHtml(ex.assistant_content || '')}</span></div>
        </div>`;
      }
    }
  } else {
    html += '<div class="ns-ledger-empty">No archived exchanges yet. Past conversations are automatically saved and indexed as you chat.</div>';
  }

  html += `</div></div>`;

  // Close ns-ltm-data-sections wrapper
  html += '</div>';

  html += '</div>';
  container.innerHTML = html;

  // Populate model dropdowns after render (async)
  _populateLtmModelDropdown();
  if (_advancedVisible) _populateAdvancedModelDropdowns();
}

// Catalog entries carry routing suffixes (``model@fabric:<node>``) that a
// stored setting may lack — the same model, two spellings. Compare on the
// base name so the stored selection still displays (2026-07-02: a bare
// stored value vs a suffixed option rendered as "Default" while the
// server used the stored model).
function _modelBase(v) {
  return String(v || '').split('@fabric:')[0];
}

function _fillModelOptions(sel, models, currentVal) {
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m.name || m.model;
    opt.textContent = m.name || m.model;
    if (opt.value === currentVal) opt.selected = true;
    sel.appendChild(opt);
  }
  if (!currentVal) return;
  const opts = Array.from(sel.options);
  if (opts.some((o) => o.selected && o.value === currentVal)) return;
  // No exact match — fall back to base-name match (first hit wins).
  const baseHit = opts.find(
    (o) => o.value && _modelBase(o.value) === _modelBase(currentVal),
  );
  if (baseHit) {
    baseHit.selected = true;
    return;
  }
  // Genuinely absent from the list: keep the stored value visible as the
  // selection instead of silently rendering the first option ("Default").
  const opt = document.createElement('option');
  opt.value = currentVal;
  opt.textContent = `${currentVal} (not in model list)`;
  opt.selected = true;
  sel.appendChild(opt);
}

async function _populateLtmModelDropdown() {
  const select = document.getElementById('ns-ltm-model');
  if (!select) return;
  const models = await getModels();
  _fillModelOptions(select, models, _ltmModel);
  select.addEventListener('change', (e) => window._nsChangeLtmModel(e.target.value));
}

async function _populateAdvancedModelDropdowns() {
  const models = await getModels();
  const fills = [
    ['ns-adv-extraction-model', _extractionModel],
    ['ns-adv-bg-distiller-model', _autoBgDistillerModel],
    ['ns-adv-bg-image-model', _autoBgImageModel],
  ];
  for (const [id, currentVal] of fills) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    _fillModelOptions(sel, models, currentVal);
  }
}

function startPolling() {
  if (pollTimer) return;
  pollNarrativeState();
  pollTimer = setInterval(pollNarrativeState, 5000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

// ---------------------------------------------------------------------------
// Character Search
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Characters / Groups Tab Bar
// ---------------------------------------------------------------------------

function initNarrativeTabBar() {
  const tabBar = document.getElementById('narrative-tab-bar');
  if (!tabBar) return;

  const contents = {
    characters: document.getElementById('narrative-tab-characters'),
    groups: document.getElementById('narrative-tab-groups'),
    persona: document.getElementById('narrative-tab-persona'),
  };

  tabBar.querySelectorAll('.narrative-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      tabBar.querySelectorAll('.narrative-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const target = tab.dataset.narrativeTab;
      for (const [key, el] of Object.entries(contents)) {
        el?.classList.toggle('hidden', key !== target);
      }
    });
  });
}

function initCharSearch() {
  const input = document.getElementById('char-search');
  if (!input) return;
  input.addEventListener('input', () => {
    const query = input.value.toLowerCase().trim();
    // Filter character wrappers
    const wrappers = document.querySelectorAll('#char-grid .char-card-wrapper');
    wrappers.forEach(wrapper => {
      const name = wrapper.querySelector('.char-name')?.textContent.toLowerCase() || '';
      wrapper.style.display = name.includes(query) || !query ? '' : 'none';
    });
    // Filter group items
    const groupItems = document.querySelectorAll('#group-list .group-item');
    groupItems.forEach(item => {
      const name = item.querySelector('.group-item-name')?.textContent.toLowerCase() || '';
      item.style.display = name.includes(query) || !query ? '' : 'none';
    });
  });
}

// ---------------------------------------------------------------------------
// Tab switching hook — update lore when tab changes
// ---------------------------------------------------------------------------

function initTabHook() {
  const select = document.getElementById('inspector-section-select');
  if (!select) return;
  select.addEventListener('change', (e) => {
    const tabId = e.target.value;
    if (tabId === 'lore-tab') {
      renderLorebook();
    } else if (tabId === 'portrait-tab') {
      updatePortraitTab();
    } else if (tabId === 'presets-tab') {
      loadPresets();
    } else if (tabId === 'regex-tab') {
      loadRegexScripts();
    } else if (tabId === 'chat-settings-tab') {
      renderNarrativeStateInspector();
    } else if (tabId === 'group-tab') {
      renderGroupInspector();
    }
  });

  // Add the group-tab option to the dropdown (hidden by default)
  const groupOpt = document.createElement('option');
  groupOpt.value = 'group-tab';
  groupOpt.textContent = 'Group';
  groupOpt.hidden = true;
  select.appendChild(groupOpt);
}

// ---------------------------------------------------------------------------
// Portrait Generator (portrait-tab)
// ---------------------------------------------------------------------------

let portraitInitialized = false;

/** Render the Auto-BG status pill — a compact summary of the active
 *  config so the user doesn't have to scan three controls to know
 *  what's on. Mounted above the section header; updated on any
 *  setting change AND on initial load. Hides when auto-bg is off. */
function _renderAutoBgPill() {
  const host = document.getElementById('visual-auto-bg-pill');
  if (!host) return;
  const check = document.getElementById('visual-auto-bg-enabled');
  if (!check || !check.checked) {
    host.innerHTML = '';
    host.classList.add('hidden');
    return;
  }
  const interval = document.getElementById('visual-auto-bg-interval');
  const imgModel = document.getElementById('visual-auto-bg-image-model');
  const intervalText = interval ? `every ${interval.value} msgs` : '';
  // "Default" maps to empty value in the select; show explicit
  // friendly text rather than blank so the pill never reads weirdly.
  const modelText = imgModel && imgModel.value
    ? imgModel.value
    : 'scene model (default)';
  const parts = ['ON'];
  if (intervalText) parts.push(intervalText);
  parts.push(modelText);
  host.innerHTML = `
    <span class="visual-auto-bg-pill-dot" aria-hidden="true"></span>
    <span class="visual-auto-bg-pill-text">${parts.map(escapeHtml).join(' · ')}</span>
  `;
  host.classList.remove('hidden');
}

async function _initVisualSceneSettings() {
  try {
    const data = await getToolSettings();

    const ctxRounds = document.getElementById('visual-scene-context-rounds');
    const autoBgCheck = document.getElementById('visual-auto-bg-enabled');
    const autoBgInterval = document.getElementById('visual-auto-bg-interval');
    const autoBgOpts = document.getElementById('visual-auto-bg-options');

    if (ctxRounds) ctxRounds.value = String(data.narrative_scene_context_rounds ?? 3);
    if (autoBgCheck) autoBgCheck.checked = data.narrative_auto_background ?? false;
    if (autoBgInterval) autoBgInterval.value = String(data.narrative_auto_background_interval ?? 4);
    if (autoBgOpts && autoBgCheck) {
      autoBgOpts.style.opacity = autoBgCheck.checked ? '1' : '0.5';
      autoBgOpts.style.pointerEvents = autoBgCheck.checked ? 'auto' : 'none';
    }

    const saveVisual = (key, value) => {
      fetch('/api/config/tools', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
      }).catch(() => {});
    };

    if (ctxRounds) ctxRounds.addEventListener('change', () => saveVisual('narrative_scene_context_rounds', parseInt(ctxRounds.value)));
    if (autoBgCheck) autoBgCheck.addEventListener('change', () => {
      saveVisual('narrative_auto_background', autoBgCheck.checked);
      if (autoBgOpts) {
        autoBgOpts.style.opacity = autoBgCheck.checked ? '1' : '0.5';
        autoBgOpts.style.pointerEvents = autoBgCheck.checked ? 'auto' : 'none';
      }
      _renderAutoBgPill();
    });
    if (autoBgInterval) autoBgInterval.addEventListener('change', () => {
      saveVisual('narrative_auto_background_interval', parseInt(autoBgInterval.value));
      _renderAutoBgPill();
    });

    // Populate model dropdowns
    const models = (await getModels()).filter(m => !m.name.startsWith('g/') && !m.name.startsWith('lb/'));
    for (const selId of ['visual-scene-distiller-model', 'visual-auto-bg-distiller']) {
      const sel = document.getElementById(selId);
      if (!sel) continue;
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        sel.appendChild(opt);
      }
    }
    try {
      const imgModels = await getImageModels();
      const imgSel = document.getElementById('visual-auto-bg-image-model');
      if (imgSel) {
        for (const m of imgModels) {
          const opt = document.createElement('option');
          opt.value = m.name;
          opt.textContent = m.name;
          imgSel.appendChild(opt);
        }
      }
    } catch { /* no image subsystem */ }

    // Set saved model values
    const distSel = document.getElementById('visual-scene-distiller-model');
    if (distSel && data.narrative_scene_distiller_model) distSel.value = data.narrative_scene_distiller_model;
    const bgDistSel = document.getElementById('visual-auto-bg-distiller');
    if (bgDistSel && data.narrative_auto_bg_distiller_model) bgDistSel.value = data.narrative_auto_bg_distiller_model;
    const bgImgSel = document.getElementById('visual-auto-bg-image-model');
    if (bgImgSel && data.narrative_auto_bg_image_model) bgImgSel.value = data.narrative_auto_bg_image_model;

    if (distSel) distSel.addEventListener('change', () => saveVisual('narrative_scene_distiller_model', distSel.value));
    if (bgDistSel) bgDistSel.addEventListener('change', () => saveVisual('narrative_auto_bg_distiller_model', bgDistSel.value));
    if (bgImgSel) bgImgSel.addEventListener('change', () => {
      saveVisual('narrative_auto_bg_image_model', bgImgSel.value);
      _renderAutoBgPill();
    });
    // First-paint render after all initial values are applied.
    _renderAutoBgPill();
  } catch { /* config not available */ }
}

function updatePortraitTab() {
  const char = getCharacter(activeCharId);
  const empty = document.getElementById('portrait-empty');
  const controls = document.getElementById('portrait-controls');
  if (!empty || !controls) return;

  // Detect group chat context
  const session = chat.getActiveSession?.();
  const isGroup = !!(session?.groupId);
  const group = isGroup ? _activeGroupForInspector : null;

  if (!char && !isGroup) {
    empty.classList.remove('hidden');
    controls.classList.add('hidden');
    return;
  }

  empty.classList.add('hidden');
  controls.classList.remove('hidden');

  // Show current avatar in preview
  const preview = document.getElementById('portrait-preview');
  if (isGroup && group?.avatar) {
    preview.innerHTML = `<img src="${escapeHtml(group.avatar)}" alt="Group Portrait">`;
  } else if (char?.avatar) {
    preview.innerHTML = `<img src="${escapeHtml(char.avatar)}" alt="Portrait">`;
  } else {
    preview.innerHTML = '<div class="portrait-placeholder">No portrait yet</div>';
  }

  // Add/remove "Group Portrait" style option AND matching chip.
  // The hidden <select> is the source of truth (existing readers
  // consume select.value); the chip is the user-facing affordance.
  const styleSelect = document.getElementById('portrait-style');
  const chipGroup = document.getElementById('portrait-style-chips');
  if (styleSelect) {
    const hasGroupOpt = !!styleSelect.querySelector('option[value="group_portrait"]');
    if (isGroup && !hasGroupOpt) {
      const opt = document.createElement('option');
      opt.value = 'group_portrait';
      opt.textContent = 'Group Portrait (all members)';
      styleSelect.insertBefore(opt, styleSelect.firstChild);
      styleSelect.value = 'group_portrait';
    } else if (!isGroup && hasGroupOpt) {
      styleSelect.querySelector('option[value="group_portrait"]').remove();
    }
  }
  if (chipGroup) {
    const groupChip = chipGroup.querySelector('.visual-chip[data-value="group_portrait"]');
    if (isGroup && !groupChip) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'visual-chip';
      chip.dataset.value = 'group_portrait';
      chip.setAttribute('role', 'radio');
      chip.innerHTML = '<span class="visual-chip-label">Group</span><span class="visual-chip-sub">all members</span>';
      chipGroup.insertBefore(chip, chipGroup.firstChild);
      // Sync active state — group_portrait was just selected above.
      for (const c of chipGroup.querySelectorAll('.visual-chip')) {
        const isActive = c.dataset.value === styleSelect.value;
        c.dataset.active = String(isActive);
        c.setAttribute('aria-checked', String(isActive));
      }
    } else if (!isGroup && groupChip) {
      groupChip.remove();
    }
  }

  if (!portraitInitialized) {
    portraitInitialized = true;
    initPortraitControls();
    _initVisualSceneSettings();
    refreshPortraitModels().then(() => {
      // Pre-select the image model from the image panel
      const imgSelect = document.getElementById('portrait-image-model');
      const panelModel = getImageSettings().model;
      if (imgSelect && panelModel) {
        // Try exact match; leave on Default if not found
        for (const opt of imgSelect.options) {
          if (opt.value === panelModel) { imgSelect.value = panelModel; break; }
        }
      }
    });
  }
}

async function refreshPortraitModels() {
  const llmSelect = document.getElementById('portrait-llm-model');
  const imgSelect = document.getElementById('portrait-image-model');

  // Populate LLM models
  if (llmSelect) {
    const models = await getModels();
    llmSelect.innerHTML = '<option value="">Default</option>';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name;
      llmSelect.appendChild(opt);
    });
  }

  // Populate image models (endpoint returns array directly, not {models: [...]})
  if (imgSelect) {
    try {
      const models = await getImageModels();
      imgSelect.innerHTML = '<option value="">Default</option>';
      (Array.isArray(models) ? models : []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name + (m.pipeline_type ? ` (${m.pipeline_type})` : '');
        if (m.is_loaded) opt.textContent += ' *';
        imgSelect.appendChild(opt);
      });
    } catch { /* image subsystem unavailable */ }
  }
}

function initPortraitControls() {
  const genPromptBtn = document.getElementById('portrait-gen-prompt-btn');
  const genImageBtn = document.getElementById('portrait-gen-image-btn');
  const useAvatarBtn = document.getElementById('portrait-use-avatar-btn');

  genPromptBtn?.addEventListener('click', handleGenPrompt);
  genImageBtn?.addEventListener('click', handleGenImage);
  useAvatarBtn?.addEventListener('click', handleUseAvatar);

  // Wire the chip group to the hidden <select> kept for backward compat.
  // Single source of truth: the hidden select. Chips push their value
  // into the select on click; existing readers (handleGenPrompt etc.)
  // see no change in API.
  const chipGroup = document.getElementById('portrait-style-chips');
  const hiddenSelect = document.getElementById('portrait-style');
  if (chipGroup && hiddenSelect) {
    const _syncChips = () => {
      const active = hiddenSelect.value;
      for (const chip of chipGroup.querySelectorAll('.visual-chip')) {
        const isActive = chip.dataset.value === active;
        chip.dataset.active = String(isActive);
        chip.setAttribute('aria-checked', String(isActive));
      }
    };
    chipGroup.addEventListener('click', (e) => {
      const chip = e.target.closest('.visual-chip');
      if (!chip || !chip.dataset.value) return;
      hiddenSelect.value = chip.dataset.value;
      // Dispatch change so any listeners on the select fire.
      hiddenSelect.dispatchEvent(new Event('change', { bubbles: true }));
      _syncChips();
    });
    // Keyboard support — Arrow keys move focus + selection, Enter confirms.
    chipGroup.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      const chips = Array.from(chipGroup.querySelectorAll('.visual-chip'));
      const current = chips.findIndex(c => c.dataset.active === 'true');
      if (current < 0) return;
      const delta = e.key === 'ArrowLeft' ? -1 : 1;
      const next = chips[(current + delta + chips.length) % chips.length];
      if (next) {
        next.click();
        next.focus();
        e.preventDefault();
      }
    });
    // Initial sync from select default (e.g. "Group Portrait" injected
    // for group sessions before the chips were built).
    _syncChips();
    // Re-sync if other code mutates the select (e.g., group-portrait
    // option injection at line 6278+).
    new MutationObserver(_syncChips).observe(hiddenSelect, {
      attributes: true, attributeFilter: ['value'], childList: true, subtree: true,
    });
    // MutationObserver doesn't fire on `select.value =` assignments —
    // observe the option list and resync from the displayed value too.
    hiddenSelect.addEventListener('change', _syncChips);
  }
}

async function handleGenPrompt() {
  const style = document.getElementById('portrait-style')?.value || 'portrait';
  const model = document.getElementById('portrait-llm-model')?.value || '';
  const loading = document.getElementById('portrait-loading');
  const promptGroup = document.getElementById('portrait-prompt-group');
  const promptTa = document.getElementById('portrait-prompt');
  const genImageBtn = document.getElementById('portrait-gen-image-btn');

  // Group portrait: send all member data
  const session = chat.getActiveSession?.();
  const isGroupPortrait = style === 'group_portrait' && session?.groupId;

  if (!isGroupPortrait) {
    const char = getCharacter(activeCharId);
    if (!char) { showToast('Select a character first', 'warning'); return; }
  }

  loading?.classList.remove('hidden');

  try {
    let body;
    if (isGroupPortrait) {
      // Build group portrait request with all member cards
      const group = _activeGroupForInspector;
      const memberCards = (group?.member_names || []).map(name => {
        const c = characters.find(ch => ch.name === name);
        return {
          name,
          visual_traits: c?.visual_traits || c?.visualTraits || '',
          description: c?.description || '',
          appearance: c?.appearance || '',
          species: c?.species || '',
        };
      });
      body = { group_members: memberCards, style: 'group_portrait', model };
    } else {
      const char = getCharacter(activeCharId);
      body = {
        name: char.name,
        description: char.description,
        personality: char.personality,
        scenario: char.scenario,
        style,
        model,
      };
    }

    const resp = await fetch('/api/ui/character-portrait-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    promptTa.value = data.prompt;
    promptGroup.style.display = '';
    genImageBtn?.classList.remove('hidden');
    autoResize(promptTa);
    showToast('Prompt generated — edit if needed, then generate image', 'success');
  } catch (err) {
    showToast('Prompt generation failed: ' + err.message, 'error');
  } finally {
    loading?.classList.add('hidden');
  }
}

async function handleGenImage() {
  const prompt = document.getElementById('portrait-prompt')?.value?.trim();
  if (!prompt) {
    showToast('Generate a prompt first', 'warning');
    return;
  }

  const staticLoading = document.getElementById('portrait-loading');
  const genImageBtn = document.getElementById('portrait-gen-image-btn');
  const useAvatarBtn = document.getElementById('portrait-use-avatar-btn');
  const preview = document.getElementById('portrait-preview');

  // Image gen swaps in the shared progress loader (stage + step counter +
  // elapsed) for the static spinner. Mounted in the same slot so the
  // visual position is unchanged; the static loader stays hidden.
  const session = chat.getActiveSession?.();
  const progress = createImageProgressLoader({
    session_id: session?.id || session?.session_id || '',
    variant: 'moment',
    category: 'user',
  });
  progress.setBaseLabel('Generating portrait…');
  staticLoading?.parentNode?.insertBefore(progress.element, staticLoading);
  progress.start();
  genImageBtn?.classList.add('hidden');

  try {
    // Inherit settings from image generation panel
    const imgPanelSettings = getImageSettings();
    // Portrait-specific model overrides panel model if set
    const portraitModel = document.getElementById('portrait-image-model')?.value || '';
    const model = portraitModel || imgPanelSettings.model || '';

    const body = {
      prompt,
      model,
      width: imgPanelSettings.width || 512,
      height: imgPanelSettings.height || 512,
    };
    // Carry over image panel settings (steps, cfg, sampler, negative, preset)
    if (imgPanelSettings.steps) body.steps = imgPanelSettings.steps;
    if (imgPanelSettings.cfg) body.cfg_scale = imgPanelSettings.cfg;
    if (imgPanelSettings.sampler) body.sampler = imgPanelSettings.sampler;
    if (imgPanelSettings.negative) body.negative_prompt = imgPanelSettings.negative;
    if (imgPanelSettings.preset) body.preset = imgPanelSettings.preset;
    if (imgPanelSettings.seed > 0) body.seed = imgPanelSettings.seed;

    const resp = await fetch('/api/image/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || err.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    const url = data.url || data.image_url;

    if (url) {
      preview.innerHTML = `<img src="${escapeHtml(url)}" alt="Generated portrait">`;
      useAvatarBtn?.classList.remove('hidden');
      showToast('Portrait generated!', 'success');
    } else {
      throw new Error('No image URL in response');
    }
  } catch (err) {
    showToast('Image generation failed: ' + err.message, 'error');
  } finally {
    progress.stop();
    genImageBtn?.classList.remove('hidden');
  }
}

async function handleUseAvatar() {
  const preview = document.getElementById('portrait-preview');
  const imgEl = preview?.querySelector('img');
  if (!imgEl) return;

  const resized = await resizeAvatar(imgEl.src);
  if (!resized) return;

  // Check if this is a group portrait context
  const style = document.getElementById('portrait-style')?.value;
  const session = chat.getActiveSession?.();
  if (style === 'group_portrait' && session?.groupId && _activeGroupForInspector) {
    // Save as group avatar
    _activeGroupForInspector.avatar = resized;
    try {
      await fetch('/api/narrative/groups', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: _activeGroupForInspector.id,
          name: _activeGroupForInspector.name,
          description: _activeGroupForInspector.description || '',
          member_names: _activeGroupForInspector.member_names,
          generation_mode: _activeGroupForInspector.generation_mode,
          member_summaries: _activeGroupForInspector.member_summaries || {},
          avatar: resized,
        }),
      });
      showToast('Group avatar updated', 'success');
      loadGroups();
    } catch { showToast('Failed to save group avatar', 'error'); }
  } else {
    // Save as character avatar
    const char = getCharacter(activeCharId);
    if (!char) return;
    char.avatar = resized;
    saveCharacters();
    renderCharGrid();
    showToast('Avatar updated from portrait', 'success');
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Reading Room Controls
// ---------------------------------------------------------------------------

function toggleReadingRoom(enabled) {
  const appEl = document.getElementById('app');
  if (!appEl) return;
  appEl.dataset.readingRoom = enabled ? 'true' : 'false';
  // Persist preference
  try { localStorage.setItem('augmentum_reading_room', enabled ? '1' : '0'); } catch {}
  // Sync toolbar button state
  const btn = document.getElementById('reading-room-btn');
  if (btn) btn.dataset.active = enabled ? 'true' : 'false';
}

function toggleReadingSepia(enabled) {
  const appEl = document.getElementById('app');
  if (!appEl) return;
  appEl.dataset.readingSepia = enabled ? 'true' : 'false';
  try { localStorage.setItem('augmentum_reading_sepia', enabled ? '1' : '0'); } catch {}
}

function toggleNarrativeBubbles(enabled) {
  const appEl = document.getElementById('app');
  if (!appEl) return;
  appEl.dataset.narrativeBubbles = enabled ? 'true' : 'false';
  try { localStorage.setItem('augmentum_narrative_bubbles', enabled ? '1' : '0'); } catch {}
}

function restoreReadingRoomState() {
  try {
    const rr = localStorage.getItem('augmentum_reading_room');
    const sepia = localStorage.getItem('augmentum_reading_sepia');
    const bubbles = localStorage.getItem('augmentum_narrative_bubbles');
    const appEl = document.getElementById('app');
    if (!appEl) return;
    if (rr === '1') appEl.dataset.readingRoom = 'true';
    if (sepia === '1') appEl.dataset.readingSepia = 'true';
    if (bubbles === '1') appEl.dataset.narrativeBubbles = 'true';
  } catch { /* ignore */ }
}

// Expose for inline event handlers
window.toggleReadingRoom = toggleReadingRoom;
window.toggleReadingSepia = toggleReadingSepia;
window.toggleNarrativeBubbles = toggleNarrativeBubbles;
window.clearNarrativeBackground = clearNarrativeBackground;
window._nsRenderLtm = renderNarrativeStateInspector;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// User Persona Management
// ---------------------------------------------------------------------------

async function loadPersonas() {
  try {
    const resp = await fetch('/api/personas/');
    if (!resp.ok) return;
    const data = await resp.json();
    personas = data.personas || [];
    renderPersonaList();
  } catch (e) {
    console.warn('Failed to load personas:', e);
  }
}

// Live-refresh narrative characters/personas when they change on the server
// (this user's other devices, via characters.changed / personas.changed on
// the SSE bus). Re-fetch + re-render only when the surface is on screen — the
// installed PWA has no manual refresh. Both listeners reference module state
// defined above; the callbacks run post-module-eval so hoisting is moot.
window.addEventListener('system-event:characters.changed', async () => {
  // A refetch replaces the whole list, so skip while local edits are unsaved
  // (mirrors the _dirtyIds guard used for sessions) to avoid clobbering work.
  if (_charDirtyIds.size > 0) return;
  const grid = document.getElementById('char-grid');
  if (!grid || grid.offsetParent === null) return;
  await loadCharacters();
  renderCharGrid();
});

window.addEventListener('system-event:personas.changed', async () => {
  const list = document.getElementById('persona-list')
    || document.getElementById('persona-list-mobile');
  if (!list || list.offsetParent === null) return;
  await loadPersonas();  // re-renders internally
});

function renderPersonaList() {
  // Update the strip display
  updatePersonaStrip();

  // Desktop (#persona-list, inside footer dropdown) + mobile (#persona-list-mobile, inside Persona tab).
  ['persona-list', 'persona-list-mobile'].forEach(id => {
    const list = document.getElementById(id);
    if (!list) return;

    if (personas.length === 0) {
      list.innerHTML = '<div class="persona-empty">No personas yet</div>';
      return;
    }

    list.innerHTML = personas.map(p => {
      const initials = (p.name || '?').slice(0, 2).toUpperCase();
      const isDefault = p.is_default;
      const classes = ['persona-item'];
      if (isDefault) classes.push('default');

      return `
        <div class="${classes.join(' ')}" data-persona-id="${escapeHtml(p.id)}">
          <div class="persona-avatar">${p.avatar ? `<img src="${escapeHtml(p.avatar)}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit">` : escapeHtml(initials)}</div>
          <div class="persona-info">
            <div class="persona-name">${escapeHtml(p.name)}${isDefault ? ' <span class="persona-badge">active</span>' : ''}</div>
          </div>
          <div class="persona-actions">
            ${!isDefault ? `<button class="icon-btn small persona-default-btn" title="Set as default" data-id="${escapeHtml(p.id)}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
            </button>` : `<button class="icon-btn small persona-undefault-btn" title="Clear default (fall back to legacy setting)" data-id="${escapeHtml(p.id)}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" fill="currentColor" fill-opacity="0.4"/><line x1="3" y1="3" x2="21" y2="21"/></svg>
            </button>`}
            <button class="icon-btn small persona-edit-btn" title="Edit" data-id="${escapeHtml(p.id)}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            </button>
            <button class="icon-btn small persona-delete-btn" title="Delete" data-id="${escapeHtml(p.id)}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </div>
        </div>
      `;
    }).join('');

    list.querySelectorAll('.persona-default-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await setDefaultPersona(btn.dataset.id);
      });
    });
    list.querySelectorAll('.persona-undefault-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await unsetDefaultPersona(btn.dataset.id);
      });
    });
    list.querySelectorAll('.persona-edit-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        openPersonaEditor(btn.dataset.id);
      });
    });
    list.querySelectorAll('.persona-delete-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await deletePersona(btn.dataset.id);
      });
    });
  });
}

function updatePersonaStrip() {
  const nameEl = document.getElementById('persona-strip-name');
  const avatarEl = document.getElementById('persona-strip-avatar');
  if (!nameEl || !avatarEl) return;

  const active = personas.find(p => p.is_default);
  if (active) {
    nameEl.textContent = active.name;
    nameEl.classList.add('active');
    if (active.avatar) {
      avatarEl.innerHTML = `<img src="${escapeHtml(active.avatar)}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit">`;
    } else {
      avatarEl.textContent = (active.name || '?').slice(0, 2).toUpperCase();
    }
  } else {
    nameEl.textContent = 'No persona';
    nameEl.classList.remove('active');
    avatarEl.innerHTML = '?';
  }
}

let personaModalEl = null;

async function refreshPersonaGenModels() {
  const select = personaModalEl?.querySelector('#persona-gen-image-model');
  if (!select) return;
  try {
    const models = await getImageModels();
    select.innerHTML = '<option value="">Default</option>';
    (Array.isArray(models) ? models : []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name + (m.pipeline_type ? ` (${m.pipeline_type})` : '');
      if (m.is_loaded) opt.textContent += ' *';
      select.appendChild(opt);
    });
  } catch { /* image subsystem unavailable */ }
}

async function handlePersonaGenAvatar() {
  const nameVal = personaModalEl.querySelector('#persona-name-input')?.value?.trim() || 'Character';
  const appearanceVal = personaModalEl.querySelector('#persona-appearance-input')?.value?.trim() || '';
  const descriptionVal = personaModalEl.querySelector('#persona-description-input')?.value?.trim() || '';

  if (!appearanceVal && !descriptionVal) {
    showToast('Add appearance or description details first', 'warning');
    return;
  }

  const style = personaModalEl.querySelector('#persona-gen-style')?.value || 'portrait';
  const imageModel = personaModalEl.querySelector('#persona-gen-image-model')?.value || '';
  const loading = personaModalEl.querySelector('#persona-gen-loading');
  const genBtn = personaModalEl.querySelector('#persona-gen-btn');

  loading?.classList.remove('hidden');
  genBtn?.classList.add('hidden');

  try {
    // Step 1: Generate image prompt from persona details via the LLM
    const promptResp = await fetch('/api/ui/character-portrait-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: nameVal,
        description: appearanceVal || descriptionVal,
        personality: '',
        scenario: '',
        style,
      }),
    });

    if (!promptResp.ok) {
      const err = await promptResp.json().catch(() => ({}));
      throw new Error(err.error || `Prompt generation failed (HTTP ${promptResp.status})`);
    }

    const { prompt } = await promptResp.json();
    if (!prompt) throw new Error('LLM returned empty prompt');

    // Step 2: Generate image from the prompt
    const imgResp = await fetch('/api/image/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        model: imageModel,
        aspect: 'square',
        width: 512,
        height: 512,
      }),
    });

    if (!imgResp.ok) {
      const err = await imgResp.json().catch(() => ({}));
      throw new Error(err.error || err.detail || `Image generation failed (HTTP ${imgResp.status})`);
    }

    const imgData = await imgResp.json();
    const imageUrl = imgData.url || imgData.image_url || '';
    if (!imageUrl) throw new Error('No image URL in response');

    // Step 3: Resize to avatar and set it
    const resized = await resizeAvatar(imageUrl);
    if (resized) {
      personaModalEl._pendingAvatar = resized;
      const preview = personaModalEl.querySelector('#persona-avatar-preview');
      preview.innerHTML = `<img src="${resized}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit">`;
      showToast('Avatar generated!', 'success');
    } else {
      throw new Error('Failed to resize generated image');
    }
  } catch (err) {
    showToast('Avatar generation failed: ' + err.message, 'error');
  } finally {
    loading?.classList.add('hidden');
    genBtn?.classList.remove('hidden');
  }
}

function openPersonaEditor(personaId = null) {
  editingPersonaId = personaId;

  if (!personaModalEl) {
    personaModalEl = document.createElement('div');
    personaModalEl.className = 'persona-modal-overlay';
    personaModalEl.innerHTML = `
      <div class="persona-modal">
        <div class="persona-modal-header">
          <span class="persona-modal-title" id="persona-modal-title">New Persona</span>
          <button class="icon-btn small" id="persona-modal-close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div class="persona-modal-body">
          <div class="field-group" style="display:flex;align-items:center;gap:var(--space-md)">
            <div class="persona-avatar-upload" id="persona-avatar-area" title="Click to upload avatar">
              <span class="persona-avatar-initials" id="persona-avatar-preview">?</span>
              <div class="persona-avatar-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" width="10" height="10"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </div>
              <input type="file" id="persona-avatar-input" accept="image/*" style="display:none">
            </div>
            <div style="flex:1">
              <label class="field-label">Name</label>
              <input class="field-input" id="persona-name-input" type="text" placeholder="Your character name...">
            </div>
          </div>
          <div class="field-group">
            <label class="field-label">Appearance</label>
            <textarea class="field-textarea" id="persona-appearance-input" rows="3" placeholder="Physical description, hair, eyes, build, clothing..."></textarea>
          </div>
          <div class="field-group">
            <label class="field-label">Description</label>
            <textarea class="field-textarea" id="persona-description-input" rows="2" placeholder="Background, personality, role..."></textarea>
          </div>

          <!-- Generate Avatar (progressive disclosure) -->
          <div class="persona-gen-section">
            <button class="btn btn-sm persona-gen-toggle" id="persona-gen-toggle">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
              Generate Avatar from Description
            </button>
            <div class="persona-gen-controls hidden" id="persona-gen-controls">
              <div class="field-group">
                <label class="field-label">Image Model</label>
                <select class="field-input" id="persona-gen-image-model">
                  <option value="">Default</option>
                </select>
              </div>
              <div class="field-group">
                <label class="field-label">Style</label>
                <select class="field-input" id="persona-gen-style">
                  <option value="portrait">Portrait</option>
                  <option value="full_body">Full Body</option>
                  <option value="anime">Anime / Manga</option>
                  <option value="photo">Photorealistic</option>
                  <option value="rpg_card">RPG / Fantasy Card</option>
                </select>
              </div>
              <button class="btn btn-primary btn-sm btn-full" id="persona-gen-btn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                Generate
              </button>
              <div class="persona-gen-loading hidden" id="persona-gen-loading">
                <div class="spinner"></div>
                <span>Generating avatar...</span>
              </div>
            </div>
          </div>
        </div>
        <div class="persona-modal-footer">
          <button class="btn" id="persona-cancel-btn">Cancel</button>
          <button class="btn btn-primary" id="persona-save-btn">Save</button>
        </div>
      </div>
    `;
    document.body.appendChild(personaModalEl);

    personaModalEl.querySelector('#persona-modal-close').addEventListener('click', closePersonaEditor);
    personaModalEl.querySelector('#persona-cancel-btn').addEventListener('click', closePersonaEditor);
    personaModalEl.querySelector('#persona-save-btn').addEventListener('click', savePersona);
    personaModalEl.addEventListener('click', (e) => {
      if (e.target === personaModalEl) closePersonaEditor();
    });

    // Avatar upload (click + drag-and-drop)
    const avatarArea = personaModalEl.querySelector('#persona-avatar-area');
    const avatarInput = personaModalEl.querySelector('#persona-avatar-input');
    avatarArea.addEventListener('click', () => avatarInput.click());
    const _applyPersonaAvatar = (resized) => {
      personaModalEl._pendingAvatar = resized;
      const preview = personaModalEl.querySelector('#persona-avatar-preview');
      preview.innerHTML = `<img src="${resized}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit">`;
    };
    avatarInput.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      const resized = await resizeAvatar(url);
      URL.revokeObjectURL(url);
      if (resized) _applyPersonaAvatar(resized);
    });
    enableAvatarDragDrop(avatarArea, _applyPersonaAvatar);

    // Generate avatar — progressive disclosure toggle
    personaModalEl.querySelector('#persona-gen-toggle').addEventListener('click', () => {
      const controls = personaModalEl.querySelector('#persona-gen-controls');
      const isHidden = controls.classList.contains('hidden');
      controls.classList.toggle('hidden');
      if (isHidden) refreshPersonaGenModels();
    });

    // Generate avatar button
    personaModalEl.querySelector('#persona-gen-btn').addEventListener('click', handlePersonaGenAvatar);

    // AI enhance buttons for persona fields
    wireEnhanceButtons(personaModalEl, 'persona', () => ({
      name: personaModalEl.querySelector('#persona-name-input')?.value || '',
      fields: {
        appearance: personaModalEl.querySelector('#persona-appearance-input')?.value || '',
        description: personaModalEl.querySelector('#persona-description-input')?.value || '',
      },
    }));
  }

  const title = personaModalEl.querySelector('#persona-modal-title');
  const nameInput = personaModalEl.querySelector('#persona-name-input');
  const appearanceInput = personaModalEl.querySelector('#persona-appearance-input');
  const descriptionInput = personaModalEl.querySelector('#persona-description-input');

  const avatarPreview = personaModalEl.querySelector('#persona-avatar-preview');
  personaModalEl._pendingAvatar = null;

  if (personaId) {
    const p = personas.find(x => x.id === personaId);
    if (!p) return;
    title.textContent = 'Edit Persona';
    nameInput.value = p.name || '';
    appearanceInput.value = p.appearance || '';
    descriptionInput.value = p.description || '';
    if (p.avatar) {
      avatarPreview.innerHTML = `<img src="${escapeHtml(p.avatar)}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit" onerror="this.remove()">`;
      personaModalEl._pendingAvatar = p.avatar;
    } else {
      avatarPreview.innerHTML = escapeHtml((p.name || '?').slice(0, 2).toUpperCase());
    }
  } else {
    title.textContent = 'New Persona';
    nameInput.value = '';
    appearanceInput.value = '';
    descriptionInput.value = '';
    avatarPreview.innerHTML = '?';
  }

  // Reset generate controls (progressive disclosure — start collapsed)
  personaModalEl.querySelector('#persona-gen-controls')?.classList.add('hidden');

  personaModalEl.style.display = '';
  nameInput.focus();
}

function closePersonaEditor() {
  if (personaModalEl) personaModalEl.style.display = 'none';
  editingPersonaId = null;
}

async function savePersona() {
  const container = personaModalEl || document;
  const nameInput = container.querySelector('#persona-name-input');
  const appearanceInput = container.querySelector('#persona-appearance-input');
  const descriptionInput = container.querySelector('#persona-description-input');

  const name = nameInput?.value?.trim();
  if (!name) {
    showToast('Name is required', 'error');
    return;
  }

  const body = {
    name,
    appearance: appearanceInput?.value?.trim() || '',
    description: descriptionInput?.value?.trim() || '',
    avatar: personaModalEl?._pendingAvatar || '',
  };

  try {
    let resp;
    if (editingPersonaId) {
      resp = await fetch(`/api/personas/${editingPersonaId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } else {
      // New persona — make default if it's the first one
      if (personas.length === 0) body.is_default = true;
      resp = await fetch('/api/personas/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    }

    if (!resp.ok) {
      const err = await resp.json();
      showToast(err.error || 'Save failed', 'error');
      return;
    }

    showToast(editingPersonaId ? 'Persona updated' : 'Persona created', 'success');
    closePersonaEditor();
    await loadPersonas();
  } catch (e) {
    showToast('Failed to save persona', 'error');
  }
}

async function deletePersona(id) {
  const p = personas.find(x => x.id === id);
  if (!p) return;
  if (!confirm(`Delete persona "${p.name}"?`)) return;

  try {
    const resp = await fetch(`/api/personas/${id}`, { method: 'DELETE' });
    if (!resp.ok) {
      showToast('Delete failed', 'error');
      return;
    }
    showToast('Persona deleted', 'success');
    await loadPersonas();
  } catch (e) {
    showToast('Failed to delete persona', 'error');
  }
}

async function setDefaultPersona(id) {
  try {
    const resp = await fetch(`/api/personas/${id}/default`, { method: 'POST' });
    if (!resp.ok) {
      showToast('Failed to set default', 'error');
      return;
    }
    showToast('Default persona updated', 'success');
    await loadPersonas();
  } catch (e) {
    showToast('Failed to set default', 'error');
  }
}

async function unsetDefaultPersona(id) {
  // Clear the default flag on this persona without picking a replacement —
  // chat falls back to the legacy `user_persona` setting until a new
  // default is set. Symmetric with setDefaultPersona above.
  try {
    const resp = await fetch(`/api/personas/${id}/undefault`, { method: 'POST' });
    if (!resp.ok) {
      showToast('Failed to clear default', 'error');
      return;
    }
    showToast('Default cleared', 'success');
    await loadPersonas();
  } catch (e) {
    showToast('Failed to clear default', 'error');
  }
}

function initPersonas() {
  loadPersonas();

  // Create persona buttons (desktop dropdown + mobile tab)
  ['create-persona-btn', 'create-persona-btn-mobile'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.addEventListener('click', () => openPersonaEditor(null));
  });

  // Persona strip toggle — open/close dropdown
  const stripToggle = document.getElementById('persona-strip-toggle');
  const dropdown = document.getElementById('persona-strip-dropdown');
  if (stripToggle && dropdown) {
    const strip = document.getElementById('persona-strip');
    stripToggle.addEventListener('click', () => {
      dropdown.classList.toggle('hidden');
      strip.classList.toggle('open');
    });
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#persona-strip')) {
        dropdown.classList.add('hidden');
        strip.classList.remove('open');
      }
    });
  }

}

// ---------------------------------------------------------------------------
// Prompt Presets (presets-tab in inspector)
// ---------------------------------------------------------------------------

let presets = [];

const MODULAR_DEFAULTS = {
  role: 'roleplayer', tense: 'present', pov: 'third',
  pov_mode: 'character', length: 'moderate', tone: 'neutral',
  content: 'sfw', anti_slop: true,
};

const MODULAR_OPTIONS = {
  role: [
    ['roleplayer', 'Roleplayer'],
    ['gm', 'Game Master'],
    ['writer', 'Writer'],
  ],
  tense: [
    ['past', 'Past'], ['present', 'Present'], ['future', 'Future'],
  ],
  pov: [
    ['first', 'First person'], ['second', 'Second person'], ['third', 'Third person'],
  ],
  pov_mode: [
    ['character', "Character's POV"], ['omniscient', 'Omniscient'],
    ['user', "User's POV"], ['flexible', 'Flexible'],
  ],
  length: [
    ['one_sentence', 'One sentence'], ['short', 'Short (~150w)'],
    ['moderate', 'Moderate (150-300w)'], ['long', 'Long (300-600w)'],
    ['chapter', 'Chapter (2000-8000w)'],
  ],
  tone: [
    ['neutral', 'Neutral'], ['expressive', 'Expressive Prose'],
    ['dialogue', 'Natural Dialogue'], ['concise', 'Concise'],
    ['cinematic', 'Cinematic'], ['slowburn', 'Slow Burn'],
  ],
  content: [
    ['sfw', 'SFW'], ['nsfw', 'NSFW'],
  ],
};

function parseModularConfig(raw) {
  if (!raw) return null;
  try {
    const cfg = JSON.parse(raw);
    return (cfg && typeof cfg === 'object') ? { ...MODULAR_DEFAULTS, ...cfg } : null;
  } catch { return null; }
}

function parseAntiSlopPhrases(raw) {
  if (!raw) return [];
  try {
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list.map(String) : [];
  } catch { return []; }
}

async function loadPresets() {
  try {
    const resp = await fetch('/api/narrative/presets');
    if (!resp.ok) return;
    const data = await resp.json();
    presets = data.presets || [];
    renderPresetList();
  } catch {
    // backend unavailable
  }
}

function renderPresetList() {
  const list = document.getElementById('presets-list');
  if (!list) return;

  if (presets.length === 0) {
    list.innerHTML = '<div class="presets-empty">No presets yet. Create one to inject prompts into your conversations.</div>';
    return;
  }

  list.innerHTML = presets.map(p => {
    const modCfg = parseModularConfig(p.modular_config);
    const isModular = !!modCfg;
    const slopPhrases = parseAntiSlopPhrases(p.anti_slop_phrases);

    const modularHtml = isModular ? `
        <div class="preset-modular-grid">
          ${Object.entries(MODULAR_OPTIONS).map(([key, opts]) => `
            <div class="field-group preset-modular-field">
              <label class="field-label">${escapeHtml(key.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()))}</label>
              <select class="field-input" data-modular="${escapeHtml(key)}">
                ${opts.map(([v, label]) =>
                  `<option value="${escapeHtml(v)}"${v === modCfg[key] ? ' selected' : ''}>${escapeHtml(label)}</option>`
                ).join('')}
              </select>
            </div>
          `).join('')}
          <div class="field-group preset-modular-field">
            <label class="field-label">Anti-Slop</label>
            <label class="preset-modular-checkbox">
              <input type="checkbox" data-modular="anti_slop"${modCfg.anti_slop ? ' checked' : ''}>
              <span>Ban cliché phrases</span>
            </label>
          </div>
        </div>
        <div class="field-group">
          <label class="field-label" title="One phrase per line. Appended to the jailbreak as a 'don't use these' directive when Anti-Slop is enabled above.">Anti-Slop Phrases <span class="field-hint-icon" title="Community-standard cliches banned by default. Add your own, one per line.">?</span></label>
          <textarea class="field-textarea" data-field="anti_slop_phrases_text" rows="4" placeholder="ministrations\nshiver down her spine\nmind, body, and soul">${escapeHtml(slopPhrases.join('\n'))}</textarea>
        </div>
        <details class="preset-modular-advanced">
          <summary>Advanced injection fields</summary>
    ` : '';

    const legacySystemField = isModular ? '' : `
        <div class="field-group">
          <label class="field-label" title="Prepended to the very beginning of the system message. Sets the overall frame for how the model should behave — writing style, roleplay rules, response format. This is the foundation that everything else builds on.">System Prompt <span class="field-hint-icon" title="Injection point 1 of 4 — highest in the context, sets the foundation">?</span></label>
          <textarea class="field-textarea" data-field="system_prompt" rows="3" placeholder="e.g. You are a skilled author collaborating on an immersive interactive story. Write vivid, evocative prose...">${escapeHtml(p.system_prompt)}</textarea>
        </div>`;

    return `
    <div class="preset-entry${p.is_default ? ' default' : ''}${isModular ? ' modular' : ''}" data-preset-id="${escapeHtml(p.id)}">
      <div class="preset-entry-header">
        <svg class="preset-entry-toggle" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
        <span class="preset-entry-name">${escapeHtml(p.name)}</span>
        ${isModular ? '<span class="preset-entry-badge modular">Modular</span>' : ''}
        ${p.is_default ? '<span class="preset-entry-badge">Default</span>' : ''}
        <div class="preset-entry-actions">
          <button class="icon-btn small" data-preset-default="${escapeHtml(p.id)}" title="Set as default">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 16.8l-6.2 4.5 2.4-7.4L2 9.4h7.6z"/></svg>
          </button>
          ${p.id.startsWith('builtin_') ? `
          <span class="icon-btn small locked" title="Built-in preset — cannot be deleted">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
          </span>
          ` : `
          <button class="icon-btn small" data-preset-delete="${escapeHtml(p.id)}" title="Delete">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
          `}
        </div>
      </div>
      <div class="preset-entry-body">
        ${modularHtml}
        ${legacySystemField}
        <div class="field-group">
          <label class="field-label" title="Injected as a system message N turns from the end of conversation. This is your primary steering wheel — models give it high priority because of its position near recent messages. Use it for style directions, scene mood, or pacing guidance.">Author's Note <span class="field-hint-icon" title="Injection point 2 of 4 — near recent messages, highest influence for steering style and tone">?</span></label>
          <textarea class="field-textarea" data-field="author_note" rows="2" placeholder="e.g. Focus on vivid sensory detail and emotional subtext. Show, don't tell.">${escapeHtml(p.author_note)}</textarea>
        </div>
        <div class="field-group">
          <label class="field-label" title="How many messages from the end of conversation to insert the Author's Note. Lower = closer to the latest messages = more influence. 4 is the community standard.">Note Depth</label>
          <div class="preset-depth-row">
            <input type="number" class="field-input" data-field="author_note_depth" value="${p.author_note_depth}" min="1" max="20" style="width:60px">
            <span class="field-hint">turns from end (4 is recommended)</span>
          </div>
        </div>
        <div class="field-group">
          <label class="field-label" title="Inserted as a system message right before the user's latest message. Good for reinforcing instructions or adding context the model should consider when responding. Less commonly used — available for your own experimentation.">Post-History <span class="field-hint-icon" title="Injection point 3 of 4 — just before the user's latest message">?</span></label>
          <textarea class="field-textarea" data-field="post_history" rows="2" placeholder="e.g. Remember to stay in character and maintain scene continuity.">${escapeHtml(p.post_history)}</textarea>
        </div>
        <div class="field-group">
          <label class="field-label" title="Inserted as a system message immediately after the user's latest message — the very last thing the model reads before generating. This position has the strongest influence on the response. Use it for character consistency rules, format enforcement, or anti-breaking-character instructions.">Jailbreak <span class="field-hint-icon" title="Injection point 4 of 4 — last thing the model sees, strongest influence on response">?</span></label>
          <textarea class="field-textarea" data-field="jailbreak" rows="2" placeholder="e.g. Stay fully immersed in the narrative. Do not break character or add meta-commentary.">${escapeHtml(p.jailbreak)}</textarea>
        </div>
        ${isModular ? '</details>' : ''}
        <div class="preset-save-row">
          <button class="btn btn-sm" data-preset-save="${escapeHtml(p.id)}">Save</button>
        </div>
      </div>
    </div>
  `;
  }).join('');

  // Wire event handlers
  list.querySelectorAll('.preset-entry-header').forEach(header => {
    header.addEventListener('click', (e) => {
      if (e.target.closest('[data-preset-default]') || e.target.closest('[data-preset-delete]') || e.target.closest('.icon-btn.locked')) return;
      header.closest('.preset-entry').classList.toggle('open');
    });
  });

  list.querySelectorAll('[data-preset-delete]').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = btn.dataset.presetDelete;
      try {
        await fetch(`/api/narrative/presets/${id}`, { method: 'DELETE' });
        showToast('Preset deleted', 'success');
        loadPresets();
      } catch { showToast('Failed to delete preset', 'error'); }
    });
  });

  list.querySelectorAll('[data-preset-default]').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = btn.dataset.presetDefault;
      const p = presets.find(x => x.id === id);
      if (!p) return;
      try {
        await fetch('/api/narrative/presets', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...p, is_default: true }),
        });
        showToast(`"${p.name}" set as default`, 'success');
        loadPresets();
      } catch { showToast('Failed to set default', 'error'); }
    });
  });

  list.querySelectorAll('[data-preset-save]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const entry = btn.closest('.preset-entry');
      const id = entry.dataset.presetId;
      const p = presets.find(x => x.id === id);
      if (!p) return;

      const isModular = entry.classList.contains('modular');
      const sysField = entry.querySelector('[data-field="system_prompt"]');

      const body = {
        id: p.id,
        name: p.name,
        is_default: p.is_default,
        // Modular presets compose their system prompt on the server from
        // modular_config, so we preserve whatever literal value was stored
        // (usually empty) rather than overwriting with an unrendered field.
        system_prompt: sysField ? sysField.value : p.system_prompt,
        author_note: entry.querySelector('[data-field="author_note"]').value,
        author_note_depth: parseInt(entry.querySelector('[data-field="author_note_depth"]').value) || 4,
        post_history: entry.querySelector('[data-field="post_history"]').value,
        jailbreak: entry.querySelector('[data-field="jailbreak"]').value,
        modular_config: p.modular_config || '',
        anti_slop_phrases: p.anti_slop_phrases || '',
      };

      if (isModular) {
        const cfg = { ...MODULAR_DEFAULTS };
        entry.querySelectorAll('[data-modular]').forEach(el => {
          const key = el.dataset.modular;
          cfg[key] = (el.type === 'checkbox') ? el.checked : el.value;
        });
        body.modular_config = JSON.stringify(cfg);

        const slopField = entry.querySelector('[data-field="anti_slop_phrases_text"]');
        if (slopField) {
          const phrases = slopField.value.split('\n').map(s => s.trim()).filter(Boolean);
          body.anti_slop_phrases = JSON.stringify(phrases);
        }
      }

      try {
        await fetch('/api/narrative/presets', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        showToast('Preset saved', 'success');
        loadPresets();
      } catch { showToast('Failed to save preset', 'error'); }
    });
  });
}

async function createPreset() {
  const name = prompt('Preset name:');
  if (!name) return;
  try {
    await fetch('/api/narrative/presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    showToast('Preset created', 'success');
    loadPresets();
  } catch { showToast('Failed to create preset', 'error'); }
}

function initPresets() {
  const addBtn = document.getElementById('preset-add-btn');
  if (addBtn) addBtn.addEventListener('click', createPreset);
}

// ---------------------------------------------------------------------------
// Regex Scripts (regex-tab in inspector)
// ---------------------------------------------------------------------------

let regexScripts = [];

async function loadRegexScripts() {
  try {
    const resp = await fetch('/api/narrative/regex');
    if (!resp.ok) return;
    const data = await resp.json();
    regexScripts = data.scripts || [];
    renderRegexList();
  } catch {
    // backend unavailable
  }
}

function renderRegexList() {
  const list = document.getElementById('regex-list');
  if (!list) return;

  if (regexScripts.length === 0) {
    list.innerHTML = '<div class="regex-empty">No regex scripts. Create scripts to transform input/output text.</div>';
    return;
  }

  list.innerHTML = regexScripts.map(s => `
    <div class="regex-entry${s.enabled ? '' : ' disabled'}" data-regex-id="${escapeHtml(s.id)}">
      <div class="regex-entry-header">
        <svg class="regex-entry-toggle" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
        <span class="regex-entry-name">${escapeHtml(s.name)}</span>
        <span class="regex-entry-placement" data-placement="${escapeHtml(s.placement)}">${escapeHtml(s.placement)}</span>
        <div class="regex-entry-enabled-toggle${s.enabled ? ' on' : ''}" data-regex-toggle="${escapeHtml(s.id)}" title="${s.enabled ? 'Disable' : 'Enable'}"></div>
        <div class="regex-entry-actions">
          <button class="icon-btn small" data-regex-delete="${escapeHtml(s.id)}" title="Delete">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
      </div>
      <div class="regex-entry-body">
        <div class="field-group">
          <label class="field-label">Name</label>
          <input class="field-input" data-field="name" value="${escapeHtml(s.name)}" placeholder="Descriptive name for this script">
        </div>
        <div class="field-group">
          <label class="field-label" title="Python-compatible regular expression. Matched text will be replaced by the replacement string below. Use capture groups like (\w+) to reference parts of the match in the replacement.">Find (regex) <span class="field-hint-icon" title="Standard regex — capture groups (\w+), alternation (a|b), quantifiers *, +, ?, character classes [a-z], etc.">?</span></label>
          <input class="field-input" data-field="find_regex" value="${escapeHtml(s.find_regex)}" style="font-family:var(--font-mono)" placeholder="e.g. \\b(orbs)\\b">
        </div>
        <div class="field-group">
          <label class="field-label" title="Text that replaces each match. Use \\1 or $1 for capture group backreferences. Use {{random:a,b,c}} to randomly pick one option per match — great for adding variety to replacements.">Replace <span class="field-hint-icon" title="Supports \\1/$1 capture groups and {{random:option1,option2}} for random selection per match">?</span></label>
          <input class="field-input" data-field="replace_string" value="${escapeHtml(s.replace_string)}" style="font-family:var(--font-mono)" placeholder="e.g. eyes">
        </div>
        <div class="lore-grid-2col">
          <div class="field-group">
            <label class="field-label" title="When this script runs in the pipeline. Input: transforms text before it reaches the model. Output: transforms the model's response before displaying. Both: runs at both stages.">Placement <span class="field-hint-icon" title="Input = before model sees text, Output = after model responds, Both = runs at both stages">?</span></label>
            <select class="field-input" data-field="placement">
              <option value="input"${s.placement === 'input' ? ' selected' : ''}>Input</option>
              <option value="output"${s.placement === 'output' ? ' selected' : ''}>Output</option>
              <option value="both"${s.placement === 'both' ? ' selected' : ''}>Both</option>
            </select>
          </div>
          <div class="field-group">
            <label class="field-label" title="Execution priority — lower numbers run first. Scripts chain together: each script's output becomes the next script's input. Preset packs use 10-56, so use 100+ for custom scripts to run after them.">Order <span class="field-hint-icon" title="Lower = runs first. Presets use 10-56; use 100+ for custom scripts">?</span></label>
            <input type="number" class="field-input" data-field="order_num" value="${s.order_num}" min="0" max="999">
          </div>
        </div>
        <div class="field-group">
          <label class="field-label" title="Scope this script to a specific character by name, or leave empty to apply globally to all characters. Character-scoped scripts only run when that character is active.">Character <span class="field-hint-icon" title="Leave empty for global (all characters). Enter a character name to limit this script to that character only.">?</span></label>
          <input class="field-input" data-field="character_name" value="${escapeHtml(s.character_name || '')}" placeholder="All characters (global)">
        </div>
        <div class="regex-save-row">
          <button class="btn btn-sm" data-regex-save="${escapeHtml(s.id)}">Save</button>
        </div>
      </div>
    </div>
  `).join('');

  // Wire event handlers
  list.querySelectorAll('.regex-entry-header').forEach(header => {
    header.addEventListener('click', (e) => {
      if (e.target.closest('[data-regex-toggle]') || e.target.closest('[data-regex-delete]')) return;
      header.closest('.regex-entry').classList.toggle('open');
    });
  });

  list.querySelectorAll('[data-regex-toggle]').forEach(toggle => {
    toggle.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = toggle.dataset.regexToggle;
      const s = regexScripts.find(x => x.id === id);
      if (!s) return;
      try {
        await fetch(`/api/narrative/regex/${id}/toggle`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: !s.enabled }),
        });
        loadRegexScripts();
      } catch { showToast('Failed to toggle', 'error'); }
    });
  });

  list.querySelectorAll('[data-regex-delete]').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = btn.dataset.regexDelete;
      try {
        await fetch(`/api/narrative/regex/${id}`, { method: 'DELETE' });
        showToast('Script deleted', 'success');
        loadRegexScripts();
      } catch { showToast('Failed to delete', 'error'); }
    });
  });

  list.querySelectorAll('[data-regex-save]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const entry = btn.closest('.regex-entry');
      const id = entry.dataset.regexId;
      const body = {
        id,
        name: entry.querySelector('[data-field="name"]').value,
        find_regex: entry.querySelector('[data-field="find_regex"]').value,
        replace_string: entry.querySelector('[data-field="replace_string"]').value,
        placement: entry.querySelector('[data-field="placement"]').value,
        order_num: parseInt(entry.querySelector('[data-field="order_num"]').value) || 100,
        character_name: entry.querySelector('[data-field="character_name"]').value || null,
        enabled: regexScripts.find(x => x.id === id)?.enabled ?? true,
      };

      try {
        const resp = await fetch('/api/narrative/regex', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const err = await resp.json();
          showToast(err.error || 'Invalid regex', 'error');
          return;
        }
        showToast('Script saved', 'success');
        loadRegexScripts();
      } catch { showToast('Failed to save', 'error'); }
    });
  });
}

async function createRegexScript() {
  const name = prompt('Script name:');
  if (!name) return;
  try {
    const resp = await fetch('/api/narrative/regex', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, find_regex: '', replace_string: '' }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      showToast(err.error || 'Failed', 'error');
      return;
    }
    showToast('Script created', 'success');
    loadRegexScripts();
  } catch { showToast('Failed to create script', 'error'); }
}

async function loadRegexPresets() {
  try {
    const resp = await fetch('/api/narrative/regex/presets');
    if (!resp.ok) return;
    const data = await resp.json();
    const list = document.getElementById('regex-presets-list');
    if (!list || !data.packs) return;

    const tierColors = { 1: '#4ade80', 2: '#60a5fa', 3: '#f59e0b', 4: '#c084fc' };

    list.innerHTML = data.packs.map(p => `
      <div class="regex-preset-card" data-preset-id="${escapeHtml(p.id)}">
        <div class="regex-preset-info">
          <span class="regex-preset-tier" style="background:${tierColors[p.tier] || '#888'}">T${p.tier}</span>
          <div class="regex-preset-text">
            <span class="regex-preset-name">${escapeHtml(p.name)}</span>
            <span class="regex-preset-desc">${escapeHtml(p.description)}</span>
          </div>
        </div>
        <button class="btn btn-sm regex-preset-install" data-install-pack="${escapeHtml(p.id)}">${p.count} scripts &mdash; Install</button>
      </div>
    `).join('');

    list.querySelectorAll('[data-install-pack]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const packId = btn.dataset.installPack;
        btn.disabled = true;
        btn.textContent = 'Installing...';
        try {
          const resp = await fetch(`/api/narrative/regex/presets/${packId}/install`, { method: 'POST' });
          const data = await resp.json();
          if (data.error) {
            showToast(data.error, 'error');
            return;
          }
          const msg = data.skipped > 0
            ? `Installed ${data.installed}, skipped ${data.skipped} (already exist)`
            : `Installed ${data.installed} scripts`;
          showToast(msg, 'success');
          loadRegexScripts();
        } catch {
          showToast('Install failed', 'error');
        } finally {
          btn.disabled = false;
          btn.textContent = btn.closest('.regex-preset-card').querySelector('.regex-preset-name').textContent.includes('Anti') ? 'Reinstall' : 'Install';
          // Restore original text
          const card = btn.closest('.regex-preset-card');
          const packData = { id: card.dataset.presetId };
          btn.innerHTML = `Install`;
        }
      });
    });
  } catch {
    // presets unavailable
  }
}

function initRegex() {
  const addBtn = document.getElementById('regex-add-btn');
  if (addBtn) addBtn.addEventListener('click', createRegexScript);
  loadRegexPresets();

  // Wire live tester
  const testBtn = document.getElementById('regex-test-btn');
  if (testBtn) {
    testBtn.addEventListener('click', async () => {
      const pattern = document.getElementById('regex-test-pattern')?.value || '';
      const replace = document.getElementById('regex-test-replace')?.value || '';
      const input = document.getElementById('regex-test-input')?.value || '';
      const resultEl = document.getElementById('regex-test-result');
      const textEl = document.getElementById('regex-test-result-text');
      const metaEl = document.getElementById('regex-test-result-meta');

      if (!pattern) {
        showToast('Enter a pattern to test', 'warning');
        return;
      }

      try {
        const resp = await fetch('/api/narrative/regex/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ find_regex: pattern, replace_string: replace, text: input }),
        });
        const data = await resp.json();
        if (data.error) {
          textEl.textContent = data.error;
          metaEl.textContent = '';
          resultEl.classList.remove('hidden');
          return;
        }
        textEl.textContent = data.result;
        metaEl.textContent = `${data.match_count} match${data.match_count !== 1 ? 'es' : ''}`;
        resultEl.classList.remove('hidden');
      } catch {
        showToast('Test failed', 'error');
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Character Groups (left panel section)
// ---------------------------------------------------------------------------

let groups = [];

async function loadGroups() {
  try {
    const resp = await fetch('/api/narrative/groups');
    if (!resp.ok) return;
    const data = await resp.json();
    groups = data.groups || [];
    renderGroupList();
    // Speaker Bar depends on the `groups` array to resolve session.groupId
    // → group. Any time groups reload, force a re-render so the bar catches
    // up (fixes "bar invisible after page refresh in a group session").
    try { renderGroupSpeakerBar(); } catch { /* not yet defined during early init */ }
  } catch {
    // backend unavailable
  }
}

function renderGroupList() {
  const list = document.getElementById('group-list');
  if (!list) return;

  // Update group count badge
  const countEl = document.getElementById('group-count');
  if (countEl) countEl.textContent = groups.length;

  if (groups.length === 0) {
    list.innerHTML = '<div class="group-empty">No groups. Create one to enable multi-character conversations.</div>';
    return;
  }

  // Build session lists per group
  const allSessions = Object.values(chat.getSessions());
  const activeSessionId = chat.getActiveSessionId();
  const groupSessionMap = {};
  for (const g of groups) {
    groupSessionMap[g.id] = allSessions
      .filter(s => s.groupId === g.id)
      .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
  }

  // Track which group is expanded (the one owning the active session, if any)
  const activeSession = activeSessionId ? allSessions.find(s => s.id === activeSessionId) : null;
  const expandedGroupId = activeSession?.groupId || null;

  list.innerHTML = '';

  for (const g of groups) {
    const chatCount = groupSessionMap[g.id].length;
    const chatLabel = chatCount === 1 ? '1 chat' : `${chatCount} chats`;
    const isExpanded = g.id === expandedGroupId;

    // Group header row
    const wrapper = document.createElement('div');
    wrapper.className = 'group-item-wrapper';

    const item = document.createElement('div');
    item.className = 'group-item' + (isExpanded ? ' expanded' : '');
    item.dataset.groupId = g.id;
    item.style.cursor = 'pointer';
    item.innerHTML = `
      <div class="group-item-icon">
        ${g.avatar
          ? `<img src="${escapeHtml(g.avatar)}" alt="" style="width:100%;height:100%;object-fit:cover;border-radius:inherit">`
          : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>`}
      </div>
      <div class="group-item-info">
        <div class="group-item-name">${escapeHtml(g.name)}</div>
        <div class="group-item-meta">${g.member_names.length} members &middot; ${escapeHtml(g.generation_mode.replace('_', ' '))}${chatCount > 0 ? ` &middot; ${chatLabel}` : ''}</div>
      </div>
      <div class="group-item-actions">
        <button class="icon-btn small" data-group-edit="${escapeHtml(g.id)}" title="Edit">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
        </button>
        <button class="icon-btn small" data-group-delete="${escapeHtml(g.id)}" title="Delete">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
    `;
    wrapper.appendChild(item);

    // Expandable chat list (reuses char-inline-chats pattern)
    if (isExpanded) {
      const accordion = document.createElement('div');
      accordion.className = 'char-inline-chats';

      let html = '';
      groupSessionMap[g.id].forEach(s => {
        const date = new Date(s.createdAt);
        const dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        const msgCount = s.tree ? Object.keys(s.tree).length : (s.messageCount || 0);
        const isCurrent = s.id === activeSessionId;
        html += `<div class="char-inline-chat${isCurrent ? ' active' : ''}" data-session-id="${escapeHtml(s.id)}" data-group-id="${escapeHtml(g.id)}">
          <svg class="char-inline-chat-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          <span class="char-inline-chat-title">${msgCount} msg${msgCount !== 1 ? 's' : ''}</span>
          <span class="char-inline-chat-date">${dateStr}</span>
          <div class="char-inline-chat-actions">
            <button class="message-action-btn" data-export-session="${escapeHtml(s.id)}" title="Export">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="10" height="10"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </button>
            <button class="message-action-btn" data-delete-session="${escapeHtml(s.id)}" title="Delete">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="10" height="10"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
        </div>`;
      });

      // New chat button
      html += `<button class="char-inline-new-chat" data-group-new-chat="${escapeHtml(g.id)}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New chat
      </button>`;

      accordion.innerHTML = html;

      // Wire chat item clicks — switch session + re-activate group
      accordion.querySelectorAll('.char-inline-chat').forEach(ci => {
        ci.addEventListener('click', (e) => {
          if (e.target.closest('[data-delete-session]') || e.target.closest('[data-export-session]')) return;
          const sid = ci.dataset.sessionId;
          const gid = ci.dataset.groupId;
          chat.switchSession(sid);
          fetch(`/api/narrative/groups/${encodeURIComponent(gid)}/activate`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid }),
          }).catch(() => {});
          renderGroupList(groups);
          renderRecentChats();
        });
      });

      accordion.querySelectorAll('[data-export-session]').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          exportNarrativeChat(btn.dataset.exportSession);
        });
      });

      accordion.querySelectorAll('[data-delete-session]').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          chat.deleteSession(btn.dataset.deleteSession);
          renderGroupList(groups);
          renderRecentChats();
        });
      });

      accordion.querySelectorAll('[data-group-new-chat]').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const gg = groups.find(x => x.id === btn.dataset.groupNewChat);
          if (gg) {
            startGroupChat(gg);
            renderRecentChats();
          }
        });
      });

      wrapper.appendChild(accordion);
    }

    // Wire header button handlers
    const editBtn = item.querySelector('[data-group-edit]');
    if (editBtn) editBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openGroupInspector(g);
    });

    const deleteBtn = item.querySelector('[data-group-delete]');
    if (deleteBtn) deleteBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      try {
        await fetch(`/api/narrative/groups/${g.id}`, { method: 'DELETE' });
        showToast('Group deleted', 'success');
        loadGroups();
        // Sessions keep their groupId for reattach-by-name (see
        // _backfillSessionGroupIds). Re-render so orphaned chips drop the
        // stale group avatar and fall back to character/placeholder.
        renderRecentChats();
      } catch { showToast('Failed to delete group', 'error'); }
    });

    // Click group header — toggle expand (show/hide chat list)
    item.addEventListener('click', () => {
      if (isExpanded) {
        // Already expanded — start new chat (no chats to pick from, or user wants fresh)
        startGroupChat(g);
      } else {
        const groupChats = groupSessionMap[g.id];
        if (groupChats.length > 0) {
          // Has chats — switch to most recent + expand
          const sid = groupChats[0].id;
          chat.switchSession(sid);
          fetch(`/api/narrative/groups/${encodeURIComponent(g.id)}/activate`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid }),
          }).catch(() => {});
          renderGroupList(groups);
          renderRecentChats();
        } else {
          // No chats — start fresh
          startGroupChat(g);
        }
      }
    });

    list.appendChild(wrapper);
  }
}

function openGroupModal(group) {
  const isNew = !group;
  const g = group || {
    name: '', description: '', member_names: [],
    generation_mode: 'round_robin', muted_names: [],
  };
  let members = [...g.member_names];
  let mutedNames = new Set(g.muted_names || []);

  const overlay = document.createElement('div');
  overlay.className = 'persona-modal-overlay';
  overlay.innerHTML = `
    <div class="persona-modal group-modal">
      <div class="persona-modal-header">
        <span class="persona-modal-title">${isNew ? 'New Group' : 'Edit Group'}</span>
        <button class="icon-btn small" id="group-modal-close">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="persona-modal-body">
        <div class="field-group">
          <label class="field-label">Name</label>
          <input class="field-input" id="group-name-input" value="${escapeHtml(g.name)}" placeholder="Group name...">
        </div>
        <div class="field-group">
          <label class="field-label">Description</label>
          <textarea class="field-textarea" id="group-desc-input" rows="2" placeholder="Optional description...">${escapeHtml(g.description)}</textarea>
        </div>
        <div class="field-group">
          <label class="field-label">Generation Mode</label>
          <select class="field-input" id="group-mode-input">
            <option value="round_robin"${g.generation_mode === 'round_robin' ? ' selected' : ''}>Round Robin</option>
            <option value="random"${g.generation_mode === 'random' ? ' selected' : ''}>Random</option>
            <option value="manual"${g.generation_mode === 'manual' ? ' selected' : ''}>Manual (click to pick)</option>
            <option value="llm_decide"${g.generation_mode === 'llm_decide' ? ' selected' : ''}>AI Director (LLM chooses)</option>
          </select>
          <div class="gi-hint" style="margin-top:4px">
            Round Robin cycles in order. Random picks (avoiding repeats). Manual uses the Speaker Bar. AI Director asks the model to pick based on recent context.
          </div>
        </div>
        <div class="field-group">
          <label class="field-label">Members</label>
          <div class="group-member-list" id="group-member-list"></div>
          <div class="group-add-member" style="margin-top:var(--space-sm)">
            <select class="field-input" id="group-add-select">
              <option value="">Add character...</option>
            </select>
            <button class="btn btn-sm" id="group-add-member-btn">Add</button>
          </div>
        </div>
      </div>
      <div class="persona-modal-footer">
        <button class="btn" id="group-modal-cancel">Cancel</button>
        <button class="btn btn-primary" id="group-modal-save">Save</button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  // Populate character select (exclude already-added members)
  function populateAddSelect() {
    const select = overlay.querySelector('#group-add-select');
    select.innerHTML = '<option value="">Add character...</option>';
    characters.forEach(c => {
      if (!members.includes(c.name)) {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.textContent = c.name;
        select.appendChild(opt);
      }
    });
  }

  function renderMembers() {
    const list = overlay.querySelector('#group-member-list');
    if (members.length === 0) {
      list.innerHTML = '<div class="group-empty">No members added yet</div>';
      return;
    }
    list.innerHTML = members.map((name, i) => {
      const isMuted = mutedNames.has(name);
      const muteTitle = isMuted
        ? 'Muted — click to unmute (still in scene, silent in rotation)'
        : 'Mute — keeps in scene context but excludes from speaking';
      return `
        <div class="group-member-item${isMuted ? ' muted' : ''}">
          <span class="group-member-name">${escapeHtml(name)}</span>
          <button class="icon-btn small group-member-mute${isMuted ? ' active' : ''}" data-mute-name="${escapeHtml(name)}" title="${muteTitle}">
            ${isMuted
              ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>'
              : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>'
            }
          </button>
          <button class="icon-btn small group-member-remove" data-remove-idx="${i}" title="Remove">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>`;
    }).join('');

    list.querySelectorAll('[data-remove-idx]').forEach(btn => {
      btn.addEventListener('click', () => {
        const removed = members[parseInt(btn.dataset.removeIdx)];
        members.splice(parseInt(btn.dataset.removeIdx), 1);
        mutedNames.delete(removed);  // keep muted set clean on remove
        renderMembers();
        populateAddSelect();
      });
    });
    list.querySelectorAll('[data-mute-name]').forEach(btn => {
      btn.addEventListener('click', () => {
        const name = btn.dataset.muteName;
        if (mutedNames.has(name)) mutedNames.delete(name);
        else mutedNames.add(name);
        renderMembers();
      });
    });
  }

  populateAddSelect();
  renderMembers();

  overlay.querySelector('#group-add-member-btn').addEventListener('click', () => {
    const select = overlay.querySelector('#group-add-select');
    const name = select.value;
    if (!name) return;
    members.push(name);
    renderMembers();
    populateAddSelect();
  });

  const close = () => overlay.remove();
  overlay.querySelector('#group-modal-close').addEventListener('click', close);
  overlay.querySelector('#group-modal-cancel').addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  overlay.querySelector('#group-modal-save').addEventListener('click', async () => {
    const name = overlay.querySelector('#group-name-input').value.trim();
    if (!name) { showToast('Name is required', 'warning'); return; }
    if (members.length < 2) { showToast('Groups need at least 2 members', 'warning'); return; }

    // Clean muted set against the final member list — removed members can't stay muted.
    const memberSet = new Set(members);
    const muted_names = [...mutedNames].filter(n => memberSet.has(n));
    const body = {
      id: g.id || undefined,
      name,
      description: overlay.querySelector('#group-desc-input').value,
      member_names: members,
      generation_mode: overlay.querySelector('#group-mode-input').value,
      muted_names,
    };

    try {
      const resp = await fetch('/api/narrative/groups', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json();
        showToast(err.error || 'Failed to save', 'error');
        return;
      }
      showToast('Group saved', 'success');
      close();
      loadGroups();
    } catch { showToast('Failed to save group', 'error'); }
  });
}

function startGroupChat(group) {
  // The lead character supplies the system-prompt scaffolding (name, persona
  // hooks, scenario). The OPENING is now chosen by the user — see
  // openGroupOpeningPicker. Free text wins over any card greeting, and there
  // is no implicit default; the picker disables Begin until the user chooses.
  const leadChar = characters.find(c => c.name === group.member_names[0]);
  if (!leadChar) { showToast('First member not found in characters', 'error'); return; }

  // Build the radio options: every member's greeting attributed by card name.
  // Members without a card or without a greeting are skipped — radios with no
  // text aren't a meaningful choice.
  const greetingOptions = [];
  for (const name of group.member_names) {
    const c = characters.find(ch => ch.name === name);
    if (c && (c.greeting || '').trim()) {
      greetingOptions.push({ name, greeting: c.greeting });
    }
  }

  openGroupOpeningPicker(group, greetingOptions, (openingText) => {
    _finalizeGroupChat(group, leadChar, openingText);
  });
}

/**
 * Finish starting a group chat with the user-chosen opening text. Split out
 * from startGroupChat so the picker callback can dispatch the same event
 * shape narrative-start-chat already consumes. The `openingText` is treated
 * as the first assistant message in the session tree (same path the existing
 * solo-chat greeting takes via tree.addChildNode in chat/index.js).
 */
function _finalizeGroupChat(group, leadChar, openingText) {
  const activePersona = personas.find(p => p.is_default);
  const userName = activePersona?.name || 'User';
  const charName = leadChar.name;
  const personaDesc = activePersona?.description || '';

  const sysParts = [`This is a group conversation with: ${group.member_names.join(', ')}.`];
  if (leadChar.systemPrompt) sysParts.push(expandCardMacros(leadChar.systemPrompt, charName, userName, personaDesc));
  if (leadChar.description) sysParts.push(expandCardMacros(leadChar.description, charName, userName, personaDesc));
  if (leadChar.personality) sysParts.push('Personality: ' + expandCardMacros(leadChar.personality, charName, userName, personaDesc));
  if (leadChar.scenario) sysParts.push('Scenario: ' + expandCardMacros(leadChar.scenario, charName, userName, personaDesc));
  for (const name of group.member_names.slice(1)) {
    const c = characters.find(ch => ch.name === name);
    if (c) {
      const brief = (c.personality || c.description || '').substring(0, 100);
      sysParts.push('[' + name + ']: ' + brief + (brief.length >= 100 ? '...' : ''));
    }
  }
  if (activePersona) {
    const pp = [];
    if (activePersona.name) pp.push('Name: ' + activePersona.name);
    if (activePersona.description) pp.push(activePersona.description);
    if (pp.length) sysParts.push('[User/Player Character]\n' + pp.join('\n'));
  }

  // Expand {{user}}/{{char}}/{{persona}} macros in whatever the user chose,
  // just like the solo-chat launchChat path does for greetings.
  const greeting = expandCardMacros(openingText, charName, userName, personaDesc);

  document.dispatchEvent(new CustomEvent('narrative-start-chat', {
    detail: {
      characterName: group.name,
      personaName: userName,
      greeting,
      systemPrompt: sysParts.join('\n\n'),
      examples: '', creatorNotes: '', lorebook: [],
      characterAvatar: leadChar.avatar || '',
      userAvatar: activePersona?.avatar || '',
      characterVoice: leadChar.voice || '',
      groupId: group.id,
      groupMembers: group.member_names,
      groupMode: group.generation_mode,
    },
  }));
}

// ---------------------------------------------------------------------------
// Group Opening Picker Modal
// ---------------------------------------------------------------------------
// Replaces the implicit first-card-wins greeting at group-chat start.
// Free-text wins; otherwise the radio-selected card greeting is used. Begin
// stays disabled until the user has actively chosen one path. Modelled on
// openGreetingPicker (same .persona-modal-* shell), but with textarea-first
// authoring as the primary path rather than alt-greeting selection.

let groupOpeningModalEl = null;

function openGroupOpeningPicker(group, greetingOptions, onChoose) {
  if (groupOpeningModalEl) groupOpeningModalEl.remove();

  const overlay = document.createElement('div');
  overlay.className = 'persona-modal-overlay';
  overlay.innerHTML = `
    <div class="persona-modal greeting-picker-modal">
      <div class="persona-modal-header">
        <span class="persona-modal-title">Start group chat: ${escapeHtml(group.name)}</span>
        <button class="icon-btn small" id="group-opening-close" title="Cancel">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="persona-modal-body">
        <div class="field-group">
          <label class="field-label" for="group-opening-freetext">Write your own opening (optional)</label>
          <textarea class="field-textarea" id="group-opening-freetext" rows="4"
            placeholder="Set the scene, describe where everyone is, or just say hi..."></textarea>
        </div>
        <div class="field-group" id="group-opening-radios-group" ${greetingOptions.length === 0 ? 'hidden' : ''}>
          <label class="field-label">Or use one of these card greetings</label>
          <div class="greeting-picker-list" id="group-opening-radios">
            ${greetingOptions.map((opt, i) => `
              <label class="greeting-picker-item" style="cursor:pointer">
                <span class="greeting-picker-label">
                  <input type="radio" name="group-opening-choice" value="${i}" style="margin-right:var(--space-xs);vertical-align:middle">
                  From ${escapeHtml(opt.name)}'s card
                </span>
                <span class="greeting-picker-preview">${escapeHtml(opt.greeting.length > 240 ? opt.greeting.slice(0, 240) + '…' : opt.greeting)}</span>
              </label>
            `).join('')}
          </div>
        </div>
      </div>
      <div class="persona-modal-footer">
        <button class="btn" id="group-opening-cancel">Cancel</button>
        <button class="btn btn-primary" id="group-opening-begin" disabled>Begin</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  groupOpeningModalEl = overlay;

  const textarea = overlay.querySelector('#group-opening-freetext');
  const beginBtn = overlay.querySelector('#group-opening-begin');
  const radios = overlay.querySelectorAll('input[name="group-opening-choice"]');

  const updateBeginEnabled = () => {
    const hasText = textarea.value.trim().length > 0;
    const hasRadio = Array.from(radios).some(r => r.checked);
    beginBtn.disabled = !(hasText || hasRadio);
  };

  textarea.addEventListener('input', () => {
    // Typing in free-text clears any radio selection so the rule
    // "free-text wins" is reflected in the UI, not just at commit.
    if (textarea.value.trim().length > 0) {
      radios.forEach(r => { r.checked = false; });
    }
    updateBeginEnabled();
  });
  radios.forEach(r => r.addEventListener('change', updateBeginEnabled));

  const close = () => {
    overlay.remove();
    if (groupOpeningModalEl === overlay) groupOpeningModalEl = null;
  };
  overlay.querySelector('#group-opening-close').addEventListener('click', close);
  overlay.querySelector('#group-opening-cancel').addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  beginBtn.addEventListener('click', () => {
    const text = textarea.value.trim();
    let opening = '';
    if (text.length > 0) {
      opening = text;
    } else {
      const checked = Array.from(radios).find(r => r.checked);
      if (checked) opening = greetingOptions[parseInt(checked.value)].greeting;
    }
    // Guard against the disabled button being bypassed (e.g. enter-key edge).
    if (!opening) { showToast('Type an opening or pick a card greeting', 'warning'); return; }
    close();
    onChoose(opening);
  });

  // Autofocus the textarea — it's the primary path.
  setTimeout(() => textarea.focus(), 50);
}

async function initGroups() {
  const createBtn = document.getElementById('create-group-btn');
  if (createBtn) createBtn.addEventListener('click', () => openGroupModal(null));
  await loadGroups();

  // Listen for session changes to update group tab visibility
  document.addEventListener('augmentum:session-changed', () => {
    updateInspectorForGroupChat();
  });
}

// ---------------------------------------------------------------------------
// Group Inspector Panel (group-tab in inspector)
// ---------------------------------------------------------------------------

let _activeGroupForInspector = null; // cached group object for the inspector

/**
 * Update inspector tab visibility based on whether the active session is a group chat.
 * In group chat: show Group tab, hide Card and Lore tabs.
 * Outside group chat: hide Group tab, show Card and Lore tabs.
 */
function updateInspectorForGroupChat() {
  const select = document.getElementById('inspector-section-select');
  if (!select) return;

  const session = chat.getActiveSession?.();
  const isGroup = !!(session?.groupId);

  // Show/hide dropdown options
  for (const opt of select.options) {
    if (opt.value === 'group-tab') {
      opt.hidden = !isGroup;
    } else if (opt.value === 'card-tab' || opt.value === 'lore-tab') {
      opt.hidden = isGroup;
    }
  }

  // If we're in a group chat and current tab is card/lore, switch to group
  if (isGroup && (select.value === 'card-tab' || select.value === 'lore-tab')) {
    app.switchInspectorSection('group-tab');
    select.value = 'group-tab';
    renderGroupInspector();
  }

  // If we left a group chat and current tab is group, switch to card
  if (!isGroup && select.value === 'group-tab') {
    app.switchInspectorSection('card-tab');
    select.value = 'card-tab';
  }
}

/**
 * Open the group inspector for a specific group (from the edit button).
 */
function openGroupInspector(group) {
  _activeGroupForInspector = group;

  // Open the inspector panel and switch to group tab
  app.switchInspectorSection('group-tab');
  const select = document.getElementById('inspector-section-select');
  if (select) {
    // Ensure group tab is visible in dropdown
    for (const opt of select.options) {
      if (opt.value === 'group-tab') opt.hidden = false;
    }
    select.value = 'group-tab';
  }

  // On mobile, open the inspector overlay
  if (window.innerWidth < 1024) {
    app.closePanel?.();
    app.openInspectorMobile();
  }

  renderGroupInspector();
}

/**
 * Render the group inspector tab content.
 */
async function renderGroupInspector() {
  const container = document.getElementById('group-tab');
  if (!container) return;

  // Resolve the group: from cached inspector state, or from active session
  let group = _activeGroupForInspector;
  if (!group) {
    const session = chat.getActiveSession?.();
    if (!session?.groupId) {
      container.innerHTML = `<div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48">
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>
          <path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
        </svg>
        <p>No group chat active. Start a group chat to see settings here.</p>
      </div>`;
      return;
    }
    // Fetch group from server
    try {
      const resp = await fetch('/api/narrative/groups');
      if (resp.ok) {
        const data = await resp.json();
        group = (data.groups || []).find(g => g.id === session.groupId);
      }
    } catch { /* ignore */ }
    if (!group) {
      container.innerHTML = '<div class="empty-state"><p>Group not found.</p></div>';
      return;
    }
    _activeGroupForInspector = group;
  }

  const summaries = group.member_summaries || {};

  let html = '<div class="group-inspector">';

  // ── Group Name ──
  html += `<div class="gi-section">
    <div class="gi-row">
      <label class="gi-label">Group Name</label>
      <input class="field-input gi-name-input" id="gi-name" value="${escapeHtml(group.name)}" placeholder="Group name...">
    </div>
  </div>`;

  // ── Turn Mode ──
  html += `<div class="gi-section">
    <div class="gi-row">
      <label class="gi-label">Turn Mode</label>
      <select class="field-input gi-mode-select" id="gi-mode">
        <option value="round_robin"${group.generation_mode === 'round_robin' ? ' selected' : ''}>Round Robin</option>
        <option value="random"${group.generation_mode === 'random' ? ' selected' : ''}>Random</option>
        <option value="manual"${group.generation_mode === 'manual' ? ' selected' : ''}>Manual</option>
        <option value="llm_decide"${group.generation_mode === 'llm_decide' ? ' selected' : ''}>AI Director</option>
      </select>
    </div>
    <div class="gi-hint">${_turnModeHint(group.generation_mode)}</div>
  </div>`;

  // ── Members + Summaries ──
  html += '<div class="gi-section"><label class="gi-label">Members &amp; Summaries</label>';

  // Get current turn state for speaker highlighting
  let currentSpeaker = '';
  const session = chat.getActiveSession?.();
  if (session?.groupId) {
    try {
      const tsResp = await fetch(`/api/narrative/groups/${encodeURIComponent(group.id)}/turn-state?session_id=${encodeURIComponent(session.id)}`);
      if (tsResp.ok) {
        const ts = await tsResp.json();
        currentSpeaker = ts.current_speaker || '';
      }
    } catch { /* ignore */ }
  }

  for (const name of group.member_names) {
    const summary = summaries[name] || '';
    const isSpeaker = name === currentSpeaker;
    const char = characters.find(c => c.name === name);
    const avatarHtml = char?.avatar
      ? `<img class="gi-member-avatar" src="${escapeHtml(char.avatar)}" alt="">`
      : `<div class="gi-member-avatar gi-member-avatar-placeholder">${escapeHtml(name.charAt(0).toUpperCase())}</div>`;
    html += `<div class="gi-member${isSpeaker ? ' speaking' : ''}" data-member-name="${escapeHtml(name)}">
      <div class="gi-member-header">
        ${avatarHtml}
        <span class="gi-member-name">${isSpeaker ? '<span class="gi-speaker-dot"></span>' : ''}${escapeHtml(name)}</span>
        <div class="gi-member-actions">
          <button class="btn btn-sm gi-generate-btn" data-gen-for="${escapeHtml(name)}" title="AI Generate Summary">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="11" height="11"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
            Generate
          </button>
          <button class="icon-btn small gi-remove-btn" data-remove-member="${escapeHtml(name)}" title="Remove from group">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="10" height="10"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
      </div>
      <textarea class="gi-summary-input" data-summary-for="${escapeHtml(name)}" rows="2" placeholder="Compact summary for other characters to see...">${escapeHtml(summary)}</textarea>
    </div>`;
  }

  // Add member
  html += `<div class="gi-add-member">
    <select class="field-input" id="gi-add-select">
      <option value="">Add character...</option>
    </select>
    <button class="btn btn-sm" id="gi-add-btn">Add</button>
  </div>`;

  html += '</div>'; // close members section

  // ── Save ──
  html += `<div class="gi-actions">
    <button class="btn btn-primary" id="gi-save-btn">Save Changes</button>
  </div>`;

  html += '</div>'; // close group-inspector

  container.innerHTML = html;

  // Populate add-character select
  const addSelect = container.querySelector('#gi-add-select');
  if (addSelect) {
    characters.forEach(c => {
      if (!group.member_names.includes(c.name)) {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.textContent = c.name;
        addSelect.appendChild(opt);
      }
    });
  }

  // ── Wire event handlers ──

  // Turn mode change — update hint and auto-save
  const modeSelect = container.querySelector('#gi-mode');
  if (modeSelect) {
    modeSelect.addEventListener('change', () => {
      const hint = container.querySelector('.gi-hint');
      if (hint) hint.textContent = _turnModeHint(modeSelect.value);
    });
  }

  // Add member
  const addBtn = container.querySelector('#gi-add-btn');
  if (addBtn) {
    addBtn.addEventListener('click', () => {
      const name = addSelect?.value;
      if (!name) return;
      group.member_names.push(name);
      group.member_summaries = group.member_summaries || {};
      _activeGroupForInspector = group;
      renderGroupInspector();
    });
  }

  // Remove member
  container.querySelectorAll('[data-remove-member]').forEach(btn => {
    btn.addEventListener('click', () => {
      const name = btn.dataset.removeMember;
      group.member_names = group.member_names.filter(n => n !== name);
      if (group.member_summaries) delete group.member_summaries[name];
      _activeGroupForInspector = group;
      renderGroupInspector();
    });
  });

  // AI Generate summary
  container.querySelectorAll('[data-gen-for]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const name = btn.dataset.genFor;
      const textarea = container.querySelector(`[data-summary-for="${CSS.escape(name)}"]`);
      if (!textarea) return;

      // Find the character card
      const char = characters.find(c => c.name === name);
      if (!char) { showToast(`Character "${name}" not found`, 'warning'); return; }

      btn.disabled = true;
      btn.textContent = 'Generating...';

      try {
        const resp = await fetch('/api/narrative/groups/generate-summary', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ character_name: name, card: char, model: app.state.currentModel || '' }),
        });
        if (resp.ok) {
          const data = await resp.json();
          textarea.value = data.summary || '';
          showToast(`Summary generated for ${name}`, 'success');
        } else {
          const err = await resp.json().catch(() => ({}));
          showToast(err.error || 'Generation failed', 'error');
        }
      } catch {
        showToast('Generation failed', 'error');
      }
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="11" height="11"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg> Generate`;
    });
  });

  // Save button
  const saveBtn = container.querySelector('#gi-save-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      // Collect current values
      const newName = container.querySelector('#gi-name')?.value?.trim() || group.name;
      const newMode = container.querySelector('#gi-mode')?.value || group.generation_mode;
      const newSummaries = {};
      container.querySelectorAll('[data-summary-for]').forEach(ta => {
        const val = ta.value.trim();
        if (val) newSummaries[ta.dataset.summaryFor] = val;
      });

      const body = {
        id: group.id,
        name: newName,
        description: group.description || '',
        member_names: group.member_names,
        generation_mode: newMode,
        member_summaries: newSummaries,
      };

      try {
        const resp = await fetch('/api/narrative/groups', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          showToast('Group saved', 'success');
          // Update cached group
          group.name = newName;
          group.generation_mode = newMode;
          group.member_summaries = newSummaries;
          _activeGroupForInspector = group;
          // Refresh the group list in the left panel
          loadGroups();
        } else {
          const err = await resp.json().catch(() => ({}));
          showToast(err.error || 'Save failed', 'error');
        }
      } catch { showToast('Save failed', 'error'); }
    });
  }
}

function _turnModeHint(mode) {
  const hints = {
    round_robin: 'Characters take turns in order (muted members are skipped). Pin a speaker from the bar to override for one turn.',
    random: 'A random unmuted character speaks each turn (avoids repeating the last speaker).',
    manual: 'You pick each next speaker from the bar above the input.',
    llm_decide: 'Before each turn, the model reads recent context and picks the most appropriate next speaker from unmuted members.',
  };
  return hints[mode] || '';
}

// ---------------------------------------------------------------------------
// JanitorAI postMessage listener (must be available immediately on page load)
// ---------------------------------------------------------------------------

let _janitorListenerRegistered = false;

async function _handleJanitorImportData(charData) {
  if (!charData || typeof charData !== 'object') {
    showToast('Invalid character data received', 'error');
    return;
  }
  try {
    const resp = await fetch('/api/characters/import-json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(charData),
    });
    const result = await resp.json();
    if (result.ok) {
      showToast('Imported: ' + (result.name || 'character'), 'success');
      await loadCharacters();
      renderCharGrid();
      if (result.id) selectCharacter(result.id);
      if (app.state.mode !== 'narrative') {
        app.dom.app.setAttribute('data-mode', 'narrative');
      }
    } else {
      showToast('Import failed: ' + (result.error || 'unknown'), 'error');
    }
  } catch (err) {
    showToast('Import failed: ' + err.message, 'error');
  }
}

function setupJanitorMessageListener() {
  if (_janitorListenerRegistered) return;
  _janitorListenerRegistered = true;
  console.debug('[Augmentum] JanitorAI postMessage listener registered');

  // Process any messages buffered by the early inline script (index.html)
  if (window.__janitorBuffer && window.__janitorBuffer.length > 0) {
    console.debug('[Augmentum] Processing', window.__janitorBuffer.length, 'buffered janitor-import messages');
    const msg = window.__janitorBuffer[window.__janitorBuffer.length - 1]; // use latest
    window.__janitorBuffer = [];
    _handleJanitorImportData(msg.data);
  }

  window.addEventListener('message', async (event) => {
    if (!event.data || event.data.type !== 'janitor-import') return;
    console.debug('[Augmentum] Received janitor-import postMessage', event.data);
    // Acknowledge so userscript stops retrying
    if (event.source) {
      try { event.source.postMessage({ type: 'janitor-ack' }, event.origin); } catch (_) { /* noop */ }
    }
    _handleJanitorImportData(event.data.data);
  });
}

// Register immediately on module load (before initNarrative is called)
setupJanitorMessageListener();

// ---------------------------------------------------------------------------
// Module Init
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tooltip positioning — uses fixed positioning to escape panel overflow
// ---------------------------------------------------------------------------
function _initTipPositioning() {
  document.addEventListener('mouseenter', (e) => {
    const tip = e.target.closest?.('.ns-tip');
    if (!tip) return;
    const rect = tip.getBoundingClientRect();
    // Position below the element, left-aligned to it but clamped to viewport
    const x = Math.max(8, Math.min(rect.left, window.innerWidth - 270));
    const y = rect.bottom + 6;
    tip.style.setProperty('--tip-x', `${x}px`);
    tip.style.setProperty('--tip-y', `${y}px`);
  }, true);
}

export async function initNarrative() {
  _initTipPositioning();
  await loadCharacters();
  // Inject character-voice lookup so TTS can resolve per-speaker voices
  // in group chats without chat/ importing narrative/ (circular dep).
  const { setCharacterVoiceLookup } = await import('../chat.js');
  setCharacterVoiceLookup((name) => {
    const lower = (name || '').toLowerCase();
    const hit = characters.find(c => (c.name || '').toLowerCase() === lower);
    return hit?.voice || '';
  });
  renderCharGrid();
  initCharSearch();
  initNarrativeTabBar();
  initTabHook();
  restoreReadingRoomState();
  initPersonas();
  initPresets();
  initRegex();
  await initGroups();
  renderRecentChats();

  // Update inspector tabs for group chat on initial load
  updateInspectorForGroupChat();

  // Re-render recent chats when a session is deleted from any source
  document.addEventListener('augmentum:session-deleted', () => {
    renderRecentChats();
    renderCharGrid();
  });

  // Re-render on active-session changes (new chat, switch) so the strip picks
  // up the newly-active chat immediately.
  document.addEventListener('augmentum:session-changed', () => {
    if (app.state.mode !== 'narrative') return;
    renderRecentChats();
    renderCharGrid();
  });

  // Re-render when session metadata updates (title generation after first
  // message, etc.) — chat.renderSessionList fires this after title mutations.
  document.addEventListener('augmentum:sessions-rendered', () => {
    if (app.state.mode !== 'narrative') return;
    renderRecentChats();
    renderCharGrid();
  });

  // Backfill legacy narrative sessions with a characterId where name is unique.
  _backfillSessionCharacterIds();
  // Legacy group sessions need groupId populated so the backend activates
  // GroupTurnManager; runs after loadGroups() has populated `groups`.
  _backfillSessionGroupIds();
  // Backfill may have stamped groupId onto the active session — re-render
  // the Speaker Bar so it appears without requiring a reload.
  try { renderGroupSpeakerBar(); } catch { /* defined later in file */ }

  // Auto-select the character that owns the active session. Only fall back
  // to characters[0] when there is NO active narrative session — otherwise
  // a failed name lookup silently mis-attributes the chat to the wrong card
  // (which then leaks into scene-image generation via activeCharacter).
  if (characters.length > 0 && !activeCharId) {
    const session = chat.getActiveSession?.();
    let picked = null;
    if (session && session.mode === 'narrative') {
      picked = _charForSession(session);
    }
    if (picked) {
      selectCharacter(picked.id);
    } else if (!session || session.mode !== 'narrative') {
      // No narrative session active — safe to show first character.
      selectCharacter(characters[0].id);
    }
    // else: active narrative session exists but owning card is missing or
    // ambiguous (e.g. duplicate names). Leave unselected so the UI shows an
    // empty state rather than silently picking the wrong card.
  } else if (characters.length === 0) {
    renderCardEditorEmpty();
  }

  // Import button — opens dialog with URL + file options
  const importBtn = document.getElementById('import-char-btn');
  if (importBtn) {
    importBtn.addEventListener('click', openImportDialog);
  }

  // Create button
  const createBtn = document.getElementById('create-char-btn');
  if (createBtn) {
    createBtn.addEventListener('click', () => openNewCharacterFlow());
  }

  // Start polling when in narrative mode
  if (app.state.mode === 'narrative') {
    startPolling();
  }

  // Listen for mode changes
  const observer = new MutationObserver(() => {
    const mode = app.dom.app.getAttribute('data-mode');
    if (mode === 'narrative') {
      startPolling();
      // Restore rotation background + frosted state when returning to narrative
      if (bgRotation.enabled && bgRotation.scope === 'narrative') {
        _applyBgDataAttrs(true);
        const mainArea = document.querySelector('.main-area');
        if (!mainArea?.querySelector('.narrative-bg-image')) {
          bgRotationNext();
        }
      }
      // Sync to active session's character if it differs
      const session = chat.getActiveSession?.();
      if (session && session.mode === 'narrative') {
        const sessionChar = _charForSession(session);
        if (sessionChar && sessionChar.id !== activeCharId) {
          selectCharacter(sessionChar.id);
        } else {
          // Restore background for active character
          const char = getCharacter(activeCharId);
          if (char) applyCharBackground(char);
        }
      } else {
        // Restore background for active character
        const char = getCharacter(activeCharId);
        if (char) applyCharBackground(char);
      }
    } else {
      stopPolling();
      applyCharBackground(null);
      // Don't clear background if rotation is active with scope 'all'
      if (!(bgRotation.enabled && bgRotation.scope === 'all')) {
        clearNarrativeBackground();
      }
    }
  });
  observer.observe(app.dom.app, { attributes: true, attributeFilter: ['data-mode'] });

  // When session changes (e.g., mode switch auto-select), sync narrative UI
  document.addEventListener('augmentum:session-changed', (e) => {
    if (app.state.mode !== 'narrative' || !e.detail?.sessionId) return;

    // Reset fingerprint so new session always gets a fresh render
    _lastStateFingerprint = '';
    // Reset LTM settings cache so per-session settings reload for the new session
    _ltmSettingsLoaded = false;

    // Immediately clear stale state from previous session before async poll
    narrativeState = null;
    renderNarrativeStateInspector();

    pollNarrativeState();

    // Auto-select the character that owns this session
    const sessions = chat.getSessions();
    const session = sessions?.[e.detail.sessionId];
    if (session && session.mode === 'narrative') {
      const matchChar = _charForSession(session);
      if (matchChar && matchChar.id !== activeCharId) {
        selectCharacter(matchChar.id);
      }
    }
  });

  // When inspector panel is opened, refresh state immediately
  document.addEventListener('augmentum:inspector-opened', () => {
    if (app.state.mode === 'narrative') pollNarrativeState();
  });

  // --- JanitorAI userscript import via URL hash ---
  // The Tampermonkey userscript intercepts JanitorAI's own API responses,
  // then opens Augmentum with #janitor-import=<encoded JSON>.
  // Also handles postMessage fallback for large payloads.
  setupJanitorMessageListener();

  if (location.hash === '#janitor-pending') {
    history.replaceState(null, '', location.pathname + location.search);
    showToast('Receiving character data from JanitorAI...', 'info', 10000);
  }

  if (location.hash.startsWith('#janitor-import=')) {
    try {
      const encoded = location.hash.substring('#janitor-import='.length);
      const json = JSON.parse(decodeURIComponent(encoded));
      history.replaceState(null, '', location.pathname + location.search);
      const result = importFromParsedJson(json);
      if (result) {
        showToast('Character imported from JanitorAI!', 'success');
        if (app.state.mode !== 'narrative') {
          app.dom.app.setAttribute('data-mode', 'narrative');
        }
      }
    } catch (err) {
      showToast('Failed to import from JanitorAI: ' + err.message, 'error');
      history.replaceState(null, '', location.pathname + location.search);
    }
  }
}

// ---------------------------------------------------------------------------
// Background Rotation Engine
// ---------------------------------------------------------------------------

const bgRotation = {
  timer: null,
  images: [],       // array of { image_id, url }
  currentIndex: -1,
  enabled: false,
  interval: 120,    // seconds
  scope: 'narrative', // 'narrative' | 'all'
  frosted: true,      // frosted glass overlay for readability
};

async function bgRotationFetchImages() {
  try {
    const resp = await fetch('/api/image/backgrounds', { credentials: 'same-origin' });
    if (!resp.ok) return [];
    const data = await resp.json();
    return (data.entries || []).map(e => ({
      image_id: e.image_id,
      url: e.url || '/api/image/' + e.image_id,
    }));
  } catch { return []; }
}

function bgRotationNext() {
  if (!bgRotation.images.length) return;

  // Never show bg rotation in coder mode
  if (app.state.mode === 'coder') return;

  // Check scope: if narrative-only, only apply when in narrative mode
  if (bgRotation.scope === 'narrative' && app.state.mode !== 'narrative') return;

  // Ensure frosted data attrs are set on every tick — they can be cleared
  // by mode switches or clearNarrativeBackground() between interval fires
  _applyBgDataAttrs(true);

  // Pick a random image (avoid repeating the same one)
  let idx;
  if (bgRotation.images.length === 1) {
    idx = 0;
  } else {
    do { idx = Math.floor(Math.random() * bgRotation.images.length); }
    while (idx === bgRotation.currentIndex);
  }
  bgRotation.currentIndex = idx;
  const img = bgRotation.images[idx];
  applyNarrativeBackground(img.url);
}

function _applyBgDataAttrs(active) {
  const appEl = document.getElementById('app');
  if (!appEl) return;
  if (active) {
    appEl.dataset.bgActive = 'true';
    appEl.dataset.bgFrosted = bgRotation.frosted ? 'true' : 'false';
  } else {
    delete appEl.dataset.bgActive;
    delete appEl.dataset.bgFrosted;
  }
}

function bgRotationStart() {
  bgRotationStop();
  if (!bgRotation.enabled || !bgRotation.images.length) return;
  _applyBgDataAttrs(true);
  // Show first image immediately
  bgRotationNext();
  bgRotation.timer = setInterval(bgRotationNext, bgRotation.interval * 1000);
}

function bgRotationStop() {
  if (bgRotation.timer) {
    clearInterval(bgRotation.timer);
    bgRotation.timer = null;
  }
  _applyBgDataAttrs(false);
}

async function bgRotationRefresh() {
  bgRotation.images = await bgRotationFetchImages();
  if (bgRotation.enabled) {
    bgRotationStart();
  }
}

export function toggleBgRotation(enabled) {
  bgRotation.enabled = enabled;
  localStorage.setItem('augmentum_bg_rotation', enabled ? '1' : '0');
  if (enabled) {
    bgRotationRefresh();
  } else {
    bgRotationStop();
    clearNarrativeBackground();
  }
}

export function setBgRotationInterval(seconds) {
  bgRotation.interval = Math.max(5, seconds);
  localStorage.setItem('augmentum_bg_rotation_interval', String(bgRotation.interval));
  if (bgRotation.enabled) bgRotationStart(); // restart with new interval
}

export function setBgRotationScope(scope) {
  bgRotation.scope = scope === 'all' ? 'all' : 'narrative';
  localStorage.setItem('augmentum_bg_rotation_scope', bgRotation.scope);
}

export function setBgRotationFrosted(frosted) {
  bgRotation.frosted = !!frosted;
  localStorage.setItem('augmentum_bg_rotation_frosted', frosted ? '1' : '0');
  _applyBgDataAttrs(bgRotation.enabled);
}

function restoreBgRotationState() {
  try {
    const enabled = localStorage.getItem('augmentum_bg_rotation') === '1';
    const interval = parseInt(localStorage.getItem('augmentum_bg_rotation_interval') || '120', 10);
    const scope = localStorage.getItem('augmentum_bg_rotation_scope') || 'narrative';
    const frosted = localStorage.getItem('augmentum_bg_rotation_frosted') !== '0'; // default true
    bgRotation.enabled = enabled;
    bgRotation.interval = Math.max(5, interval);
    bgRotation.scope = scope;
    bgRotation.frosted = frosted;
    if (enabled) bgRotationRefresh();
  } catch { /* ignore */ }
}

// Expose for cross-module use
window.toggleBgRotation = toggleBgRotation;
window.setBgRotationInterval = setBgRotationInterval;
window.setBgRotationScope = setBgRotationScope;
window.setBgRotationFrosted = setBgRotationFrosted;
window._bgRotationRefresh = bgRotationRefresh;
window._bgRotationState = bgRotation;

// Re-apply background on mode switch when scope is 'all'
document.addEventListener('augmentum:mode-changed', () => {
  if (bgRotation.enabled && bgRotation.scope === 'all') {
    _applyBgDataAttrs(true);
    // Immediately show a background if none is visible
    const mainArea = document.querySelector('.main-area');
    if (!mainArea?.querySelector('.narrative-bg-image')) {
      bgRotationNext();
    }
  }
});

// Restore on load (called from initNarrative)
restoreBgRotationState();

// ---------------------------------------------------------------------------
// Group Speaker Bar — per-turn speaker control for group chats.
//
// Renders above the composer in group sessions: an avatar strip showing
// each member, with visual state for the predicted-next speaker (breathing
// ring), muted members (dimmed + 🔇 badge), pinned speaker (solid glow +
// lock badge), and an LLM-thinking spinner when mode=llm_decide.
//
// Interactions:
//   - Click avatar           → pin as next speaker (release after turn)
//   - Click pinned avatar    → unpin, back to auto
//   - Right-click avatar     → context menu: mute/unmute
//   - Slash commands         → /as Name, /mute Name, /unmute Name, /unpin
// ---------------------------------------------------------------------------

const SPEAKER_BAR_ID = 'group-speaker-bar';

function _activeNarrativeSession() {
  const s = chat.getActiveSession?.();
  return s && s.mode === 'narrative' ? s : null;
}

function _groupForSession(session) {
  if (!session?.groupId) return null;
  return groups.find(g => g.id === session.groupId) || null;
}

function _ensureSpeakerBarMount() {
  let bar = document.getElementById(SPEAKER_BAR_ID);
  if (bar) return bar;
  const area = document.getElementById('chat-input')?.closest('.input-area');
  if (!area) return null;
  bar = document.createElement('div');
  bar.id = SPEAKER_BAR_ID;
  bar.className = 'group-speaker-bar';
  bar.setAttribute('role', 'toolbar');
  bar.setAttribute('aria-label', 'Group chat speaker');
  // Mount as the first child of .input-area so it sits above the toolbar.
  area.insertBefore(bar, area.firstChild);
  return bar;
}

function _predictedNextSpeaker(session, group) {
  // Manual pin wins.
  if (session?.nextSpeaker) return session.nextSpeaker;
  // llm_decide → shown as "AI will choose" until the actual call resolves.
  if (group?.generation_mode === 'llm_decide') return null;
  // manual mode (no pin) → no prediction; user must choose.
  if (group?.generation_mode === 'manual') return null;
  // round_robin / random: show the most recent tracked speaker's next.
  // We don't have perfect knowledge client-side; best-effort uses the
  // stored group_speaker_index via the last assistant node, else first.
  const lastNode = _lastAssistantSpeakerName(session);
  if (lastNode) {
    const unmuted = (group?.member_names || []).filter(
      n => !(group?.muted_names || []).includes(n),
    );
    if (unmuted.length <= 1) return unmuted[0] || null;
    if (group.generation_mode === 'random') return null;  // can't predict random
    // round_robin: next after lastNode in member_names (skip muted)
    const order = group.member_names || [];
    const lastIdx = order.findIndex(n => n === lastNode);
    for (let step = 1; step <= order.length; step++) {
      const cand = order[(lastIdx + step) % order.length];
      if (!(group.muted_names || []).includes(cand)) return cand;
    }
  }
  // First unmuted member otherwise
  return (group?.member_names || []).find(
    n => !(group?.muted_names || []).includes(n),
  ) || null;
}

function _lastAssistantSpeakerName(session) {
  if (!session?.tree || !session.activeLeafId) return null;
  let id = session.activeLeafId;
  while (id) {
    const node = session.tree[id];
    if (!node) break;
    if (node.role === 'assistant' && node.speakerName) return node.speakerName;
    id = node.parentId;
  }
  return null;
}

function _muteIconSvg() {
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>';
}

function _lockIconSvg() {
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" width="10" height="10"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>';
}

function _spinnerSvg() {
  return '<svg class="gsb-spinner" viewBox="0 0 24 24" width="14" height="14"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2.2" stroke-dasharray="14 14" stroke-linecap="round"/></svg>';
}

const GSB_STATE_KEY = 'augmentum_speaker_bar_minimized';

function _gsbMinimized() {
  try { return localStorage.getItem(GSB_STATE_KEY) === '1'; }
  catch { return false; }
}
function _gsbSetMinimized(v) {
  try { localStorage.setItem(GSB_STATE_KEY, v ? '1' : '0'); } catch {}
}

function _chevronIconSvg(down) {
  const d = down ? 'M6 9l6 6 6-6' : 'M18 15l-6-6-6 6';
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="${d}"/></svg>`;
}

function renderGroupSpeakerBar() {
  const bar = _ensureSpeakerBarMount();
  if (!bar) return;

  const session = _activeNarrativeSession();
  const group = _groupForSession(session);

  if (!session || !group || app.state.mode !== 'narrative') {
    bar.hidden = true;
    bar.innerHTML = '';
    return;
  }

  bar.hidden = false;
  bar.classList.toggle('minimized', _gsbMinimized());

  // Minimized: collapsed pill on the left showing just the "next speaker"
  // initial — one click expands back to full bar.
  if (_gsbMinimized()) {
    const predicted = _predictedNextSpeaker(session, group);
    const pinned = session.nextSpeaker || '';
    const who = pinned || predicted;
    const char = who ? characters.find(c => c.name === who) : null;
    const dot = char?.avatar
      ? `<img class="gsb-mini-avatar" src="${escapeHtml(char.avatar)}" alt="${escapeHtml(who || '')}">`
      : `<span class="gsb-mini-initial">${escapeHtml((who || group.name || '?').charAt(0).toUpperCase())}</span>`;
    const labelHtml = pinned
      ? `<strong>${escapeHtml(pinned)}</strong> pinned`
      : (who ? `Next: <strong>${escapeHtml(who)}</strong>` : escapeHtml(group.name));
    bar.innerHTML = `
      <button type="button" class="gsb-mini" title="Expand speaker bar">
        ${dot}
        <span class="gsb-mini-label">${labelHtml}</span>
        ${_chevronIconSvg(false)}
      </button>`;
    bar.querySelector('.gsb-mini')?.addEventListener('click', () => {
      _gsbSetMinimized(false);
      renderGroupSpeakerBar();
    });
    return;
  }

  // Expanded compact layout — single row: avatars + inline label + minimize chevron.
  const predicted = _predictedNextSpeaker(session, group);
  const pinned = session.nextSpeaker || '';
  const mode = group.generation_mode || 'round_robin';
  const muted = group.muted_names || [];

  let statusHtml;
  if (pinned) {
    statusHtml = `<strong>${escapeHtml(pinned)}</strong> <span class="gsb-mode-tag">pinned</span>`;
  } else if (mode === 'llm_decide') {
    statusHtml = `<span class="gsb-llm-tag">AI</span> picking…`;
  } else if (mode === 'manual') {
    statusHtml = `<span class="gsb-mode-tag">manual</span> pick a speaker`;
  } else if (predicted) {
    statusHtml = `<strong>${escapeHtml(predicted)}</strong> <span class="gsb-mode-tag">${mode === 'random' ? 'random' : 'rotation'}</span>`;
  } else {
    statusHtml = `<span class="gsb-mode-tag">${escapeHtml(mode)}</span>`;
  }

  let avatarsHtml = '';
  for (const name of group.member_names) {
    const char = characters.find(c => c.name === name);
    const avatar = char?.avatar;
    const isMuted = muted.includes(name);
    const isPinned = pinned === name;
    const isPredicted = !pinned && predicted === name;
    const classes = [
      'gsb-avatar',
      isMuted ? 'muted' : '',
      isPinned ? 'pinned' : '',
      isPredicted ? 'predicted' : '',
    ].filter(Boolean).join(' ');
    const avatarInner = avatar
      ? `<img src="${escapeHtml(avatar)}" alt="${escapeHtml(name)}">`
      : `<span class="gsb-initial">${escapeHtml(name.charAt(0).toUpperCase())}</span>`;
    const badge = isPinned
      ? `<span class="gsb-badge pin" title="Pinned as next speaker">${_lockIconSvg()}</span>`
      : (isMuted ? `<span class="gsb-badge mute" title="Muted">${_muteIconSvg()}</span>` : '');
    const tooltip = isMuted
      ? `${name} — muted (right-click to unmute)`
      : (isPinned ? `Unpin ${name}` : `Pin ${name} to speak next`);
    avatarsHtml += `
      <button type="button" class="${classes}" data-speaker="${escapeHtml(name)}" title="${escapeHtml(tooltip)}">
        ${avatarInner}${badge}
      </button>`;
  }

  bar.innerHTML = `
    <div class="gsb-avatars">${avatarsHtml}</div>
    <div class="gsb-status">${statusHtml}</div>
    <button type="button" class="gsb-collapse" title="Minimize speaker bar">
      ${_chevronIconSvg(true)}
    </button>`;

  bar.querySelectorAll('.gsb-avatar').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      _togglePinSpeaker(btn.dataset.speaker);
    });
    btn.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      _showSpeakerContextMenu(e, btn.dataset.speaker);
    });
  });
  bar.querySelector('.gsb-collapse')?.addEventListener('click', () => {
    _gsbSetMinimized(true);
    renderGroupSpeakerBar();
  });
}

function _togglePinSpeaker(name) {
  const session = _activeNarrativeSession();
  if (!session) return;
  if (session.nextSpeaker === name) {
    session.nextSpeaker = '';
    showToast(`Unpinned — back to auto`, 'info');
  } else {
    session.nextSpeaker = name;
    showToast(`${name} will speak next`, 'info');
  }
  chat.saveSessions?.(session.id);
  renderGroupSpeakerBar();

  // Mirror the pin to the server-side `force_speaker` slot on the group's
  // turn-state. Without this, the LLM-decide path doesn't see the user's
  // pin and may still pick a different speaker; with it, the next turn
  // honors the override even after a page reload.
  const group = _groupForSession(session);
  if (group?.id) {
    fetch(`/api/narrative/groups/${encodeURIComponent(group.id)}/force-speaker`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        session_id: session.id,
        speaker_name: session.nextSpeaker || null,
      }),
    }).catch(() => { /* fire-and-forget; client-side pin still drives UI */ });
  }
}

let _gsbCtxMenu = null;

function _showSpeakerContextMenu(event, name) {
  _closeSpeakerContextMenu();
  const session = _activeNarrativeSession();
  const group = _groupForSession(session);
  if (!session || !group) return;

  const isMuted = (group.muted_names || []).includes(name);
  const menu = document.createElement('div');
  menu.className = 'gsb-ctx-menu';
  menu.innerHTML = `
    <button data-action="pin">${session.nextSpeaker === name ? 'Unpin' : 'Pin as next speaker'}</button>
    <button data-action="mute">${isMuted ? 'Unmute' : 'Mute'}</button>
  `;
  document.body.appendChild(menu);
  // Position near the click, keeping inside viewport
  const rect = menu.getBoundingClientRect();
  const x = Math.min(event.clientX, window.innerWidth - rect.width - 8);
  const y = Math.max(8, event.clientY - rect.height - 8);
  menu.style.left = x + 'px';
  menu.style.top = y + 'px';
  _gsbCtxMenu = menu;

  menu.addEventListener('click', async (e) => {
    const action = e.target.closest('[data-action]')?.dataset.action;
    _closeSpeakerContextMenu();
    if (action === 'pin') {
      _togglePinSpeaker(name);
    } else if (action === 'mute') {
      await _setMemberMuted(group, name, !isMuted);
    }
  });
  // Dismiss on outside click or Escape
  setTimeout(() => {
    document.addEventListener('click', _closeSpeakerContextMenu, { once: true, capture: true });
    document.addEventListener('keydown', _speakerCtxKey, { once: true });
  }, 0);
}

function _closeSpeakerContextMenu() {
  if (_gsbCtxMenu) {
    _gsbCtxMenu.remove();
    _gsbCtxMenu = null;
  }
}

function _speakerCtxKey(e) {
  if (e.key === 'Escape') _closeSpeakerContextMenu();
}

async function _setMemberMuted(group, name, muted) {
  const muted_names = new Set(group.muted_names || []);
  if (muted) muted_names.add(name); else muted_names.delete(name);
  const next = {
    id: group.id,
    name: group.name,
    description: group.description,
    member_names: group.member_names,
    generation_mode: group.generation_mode,
    member_summaries: group.member_summaries,
    avatar: group.avatar,
    muted_names: [...muted_names],
  };
  try {
    const resp = await fetch('/api/narrative/groups', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(next),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    // Update local cache
    group.muted_names = next.muted_names;
    showToast(`${name} ${muted ? 'muted' : 'unmuted'}`, 'success');
    renderGroupSpeakerBar();
    renderRecentChats();
  } catch (err) {
    showToast(`Failed to update: ${err.message}`, 'error');
  }
}

// --- Slash commands in the chat composer -----------------------------------

function _findMemberLoose(group, query) {
  if (!group || !query) return null;
  const q = query.trim().toLowerCase();
  return group.member_names.find(n => n.toLowerCase() === q)
      || group.member_names.find(n => n.toLowerCase().startsWith(q))
      || null;
}

async function _handleGroupSlashCommand(textarea) {
  const raw = textarea.value.trim();
  if (!raw.startsWith('/')) return false;
  const session = _activeNarrativeSession();
  const group = _groupForSession(session);
  if (!session || !group) return false;

  const [cmd, ...rest] = raw.split(/\s+/);
  const arg = rest.join(' ').trim();
  const lc = cmd.toLowerCase();

  if (lc === '/as') {
    const target = _findMemberLoose(group, arg);
    if (!target) {
      showToast(`Unknown member: ${arg || '(none)'}`, 'warning');
      return true;
    }
    session.nextSpeaker = target;
    chat.saveSessions?.(session.id);
    textarea.value = '';
    renderGroupSpeakerBar();
    showToast(`${target} pinned — send a message`, 'info');
    return true;
  }
  if (lc === '/unpin') {
    session.nextSpeaker = '';
    chat.saveSessions?.(session.id);
    textarea.value = '';
    renderGroupSpeakerBar();
    showToast('Unpinned — back to auto', 'info');
    return true;
  }
  if (lc === '/mute' || lc === '/unmute') {
    const target = _findMemberLoose(group, arg);
    if (!target) {
      showToast(`Unknown member: ${arg || '(none)'}`, 'warning');
      return true;
    }
    textarea.value = '';
    await _setMemberMuted(group, target, lc === '/mute');
    return true;
  }
  return false;
}

function _wireGroupSlashCommands() {
  const textarea = document.getElementById('chat-input');
  if (!textarea || textarea._gsbWired) return;
  textarea._gsbWired = true;
  // Capture phase so we preempt the ChatInput's own keydown handler.
  textarea.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter' || e.shiftKey) return;
    const handled = await _handleGroupSlashCommand(textarea);
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, { capture: true });
}

// Keep the bar in sync with session / group / mode changes.
document.addEventListener('augmentum:session-changed', () => renderGroupSpeakerBar());
document.addEventListener('augmentum:mode-changed', () => renderGroupSpeakerBar());
document.addEventListener('augmentum:speaker-pin-released', () => renderGroupSpeakerBar());

// Voice command "chat with <name>" — resolve the character by fuzzy
// name match and hand off to the same launch path the picker uses.
// The matcher is case-insensitive and prefers startsWith over substring,
// so "sam" hits "Samantha" before "Prince Adam" if both are loaded.
document.addEventListener('voice:find-character', (e) => {
  const name = (e.detail?.name || '').trim().toLowerCase();
  if (!name) return;
  const pool = Array.isArray(characters) ? characters : [];
  const starts = pool.find(c => (c?.name || '').toLowerCase().startsWith(name));
  const contains = starts || pool.find(c => (c?.name || '').toLowerCase().includes(name));
  if (contains) startChatWithCharacter(contains);
});

// Initial wire-up. ES modules can execute AFTER DOMContentLoaded has fired,
// so run immediately when the DOM is already past 'loading' and fall back to
// the event listener otherwise. Fixes "bar invisible on hard reload".
function _gsbBootstrap() {
  _wireGroupSlashCommands();
  renderGroupSpeakerBar();
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _gsbBootstrap);
} else {
  _gsbBootstrap();
}

export const narrative = {
  createCharacter,
  deleteCharacter,
  importFromUrl,
  importCharacterFromFile,
  selectCharacter,
  openImportDialog,
  startChatWithCharacter,
  renderGroupSpeakerBar,
  get characters() { return characters; },
  get activeCharId() { return activeCharId; },
  get activeCharacter() { return getCharacter(activeCharId); },
};
