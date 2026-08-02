// Save-to-Library save prompt anchored to the coder preview pane.
//
// Exports a single function — openSavePrompt — that the preview-pane
// save button calls. The prompt is a small anchored popover (not a
// modal) with the title input + a derived summary. Title-collision
// detection is debounced against the preflight endpoint so the
// overwrite-or-rename split surfaces before the user clicks Save.
//
// Preview state (served dir / size / kind) is server-side authoritative
// — we just present what the preflight returns, never derive it
// client-side.

import { escapeHtml, showToast } from './app.js';

// Debounce window for title-typing → preflight collision check. Long
// enough to avoid hammering the route during a multi-keystroke type,
// short enough that the warning surfaces before the user clicks Save.
const _COLLISION_DEBOUNCE_MS = 350;

// Module state — only one save prompt can be open at a time. Cached so
// the close handler can revert event listeners + restore focus.
let _activePrompt = null;

function _fmtBytes(n) {
  if (!n || n < 1024) return `${n || 0} B`;
  const units = ['KB', 'MB', 'GB'];
  let v = n / 1024, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

function _fmtRelativeTime(epochSeconds) {
  if (!epochSeconds) return '';
  const now = Date.now() / 1000;
  const delta = now - epochSeconds;
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)} h ago`;
  return `${Math.floor(delta / 86400)} d ago`;
}

async function _preflight(workspaceId, proposedTitle = '') {
  const resp = await fetch('/api/library/save/preflight', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({
      workspace_id: workspaceId,
      proposed_title: proposedTitle,
    }),
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    const err = new Error(`preflight failed: ${resp.status} ${text}`);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}

async function _save({ workspaceId, title, description, onCollision }) {
  const resp = await fetch('/api/library/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({
      workspace_id: workspaceId,
      title,
      description,
      on_collision: onCollision,
    }),
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(body.hint || body.error || `save failed: ${resp.status}`);
    err.status = resp.status;
    err.body = body;
    throw err;
  }
  return body;
}

function _renderPromptShell(promptEl, snap) {
  const fileCount = snap.file_count || 0;
  const sizeLabel = _fmtBytes(snap.estimated_size_bytes);
  promptEl.innerHTML = `
    <div class="coder-save-prompt-inner">
      <div class="coder-save-prompt-header">
        <span class="coder-save-prompt-title">Save to Library</span>
        <button class="coder-save-prompt-close" id="coder-save-close" title="Cancel" aria-label="Cancel">✕</button>
      </div>
      <label class="coder-save-prompt-field">
        <span>Title</span>
        <input type="text" id="coder-save-title-input" maxlength="160" autocomplete="off" placeholder="My game">
      </label>
      <label class="coder-save-prompt-field">
        <span>Description <span class="coder-save-prompt-hint">optional</span></span>
        <input type="text" id="coder-save-desc-input" maxlength="200" placeholder="One-line summary">
      </label>
      <div class="coder-save-prompt-meta">
        Includes ${fileCount} file${fileCount === 1 ? '' : 's'} · ${escapeHtml(sizeLabel)}
      </div>
      <div class="coder-save-prompt-warning hidden" id="coder-save-warning"></div>
      <div class="coder-save-prompt-actions">
        <button class="btn-secondary small" id="coder-save-cancel">Cancel</button>
        <button class="btn-primary small" id="coder-save-submit">Save</button>
      </div>
    </div>
  `;
}

function _renderRefusal(promptEl, snap) {
  const kind = snap.preview_kind || 'none';
  let title = 'No preview running';
  let body = 'Start a static service first (python -m http.server, npx serve, vite preview) and confirm the preview pane loads it. Then try again.';
  if (kind === 'dynamic') {
    title = 'Dynamic preview not supported yet';
    body = 'This preview is served by a runtime (uvicorn, node server, etc.). Static save is v1 only. To save anyway, switch to a static build: <code>npm run build && npx serve dist</code>, then retry.';
  } else if (kind === 'unknown') {
    title = "Couldn't classify the preview";
    body = 'The probe didn’t get a clean response. Confirm the dev server is healthy in the preview pane, then retry.';
  }
  promptEl.innerHTML = `
    <div class="coder-save-prompt-inner">
      <div class="coder-save-prompt-header">
        <span class="coder-save-prompt-title">${escapeHtml(title)}</span>
        <button class="coder-save-prompt-close" id="coder-save-close" title="Close" aria-label="Close">✕</button>
      </div>
      <div class="coder-save-prompt-refusal">${body}</div>
    </div>
  `;
}

function _showCollisionWarning(promptEl, existing) {
  const warn = promptEl.querySelector('#coder-save-warning');
  if (!warn) return;
  const rel = existing.existing_updated_at
    ? _fmtRelativeTime(existing.existing_updated_at)
    : '';
  warn.classList.remove('hidden');
  warn.innerHTML = `
    A publication with this title already exists${rel ? ` (last saved ${escapeHtml(rel)})` : ''}.
    Choose <em>Overwrite</em> to replace it, or pick a different title.
  `;
  // Mutate the submit button to make overwrite explicit.
  const submit = promptEl.querySelector('#coder-save-submit');
  if (submit) {
    submit.textContent = 'Overwrite →';
    submit.dataset.collision = '1';
  }
}

function _clearCollisionWarning(promptEl) {
  const warn = promptEl.querySelector('#coder-save-warning');
  if (warn) { warn.classList.add('hidden'); warn.innerHTML = ''; }
  const submit = promptEl.querySelector('#coder-save-submit');
  if (submit) { submit.textContent = 'Save'; delete submit.dataset.collision; }
}

function _closePrompt() {
  if (!_activePrompt) return;
  const { promptEl, outsideHandler, keyHandler } = _activePrompt;
  promptEl.classList.add('hidden');
  promptEl.innerHTML = '';
  document.removeEventListener('mousedown', outsideHandler, true);
  document.removeEventListener('keydown', keyHandler, true);
  _activePrompt = null;
}

function _onOutside(event) {
  if (!_activePrompt) return;
  const { promptEl, anchorEl } = _activePrompt;
  if (promptEl.contains(event.target)) return;
  if (anchorEl && anchorEl.contains(event.target)) return;
  _closePrompt();
}

function _onKey(event) {
  if (event.key === 'Escape') _closePrompt();
}

/**
 * Open the save prompt anchored to `anchorEl`, populated from a
 * preflight call against `workspaceId`. The prompt is single-active
 * (calling again closes any prior prompt).
 *
 * @param {string} workspaceId  Coder workspace id.
 * @param {HTMLElement} anchorEl  The button the prompt anchors near.
 * @param {HTMLElement} promptEl  The container div (already in DOM,
 *                                styled by coder.css, hidden by default).
 */
export async function openSavePrompt(workspaceId, anchorEl, promptEl) {
  if (!workspaceId || !promptEl) return;
  if (_activePrompt) _closePrompt();

  promptEl.classList.remove('hidden');
  promptEl.innerHTML = '<div class="coder-save-prompt-loading">Checking preview…</div>';

  let snap;
  try {
    snap = await _preflight(workspaceId);
  } catch (exc) {
    promptEl.classList.add('hidden');
    promptEl.innerHTML = '';
    if (exc.status === 401) showToast('Auth required', 'error');
    else if (exc.status === 404) showToast('Workspace not found', 'error');
    else showToast('Preflight failed', 'error');
    return;
  }

  if (!snap.preview_ready) {
    _renderRefusal(promptEl, snap);
    const close = promptEl.querySelector('#coder-save-close');
    close?.addEventListener('click', _closePrompt);
    _activePrompt = {
      promptEl,
      anchorEl,
      outsideHandler: _onOutside,
      keyHandler: _onKey,
    };
    document.addEventListener('mousedown', _onOutside, true);
    document.addEventListener('keydown', _onKey, true);
    return;
  }

  _renderPromptShell(promptEl, snap);
  const titleInput = promptEl.querySelector('#coder-save-title-input');
  const descInput = promptEl.querySelector('#coder-save-desc-input');
  const cancelBtn = promptEl.querySelector('#coder-save-cancel');
  const closeBtn = promptEl.querySelector('#coder-save-close');
  const submitBtn = promptEl.querySelector('#coder-save-submit');

  _activePrompt = {
    promptEl,
    anchorEl,
    outsideHandler: _onOutside,
    keyHandler: _onKey,
  };
  document.addEventListener('mousedown', _onOutside, true);
  document.addEventListener('keydown', _onKey, true);

  cancelBtn?.addEventListener('click', _closePrompt);
  closeBtn?.addEventListener('click', _closePrompt);

  // Debounced title-collision check on input. Server is authoritative;
  // the check is purely a UX hint so users see overwrite-vs-rename
  // before clicking Save.
  let debounceHandle = null;
  let lastChecked = '';
  titleInput?.addEventListener('input', () => {
    if (debounceHandle) clearTimeout(debounceHandle);
    const proposed = titleInput.value.trim();
    debounceHandle = setTimeout(async () => {
      if (proposed === lastChecked) return;
      lastChecked = proposed;
      if (!proposed) { _clearCollisionWarning(promptEl); return; }
      try {
        const probe = await _preflight(workspaceId, proposed);
        if (probe.title_collision) _showCollisionWarning(promptEl, probe);
        else _clearCollisionWarning(promptEl);
      } catch {
        // Ignore — best-effort hint. Server enforces on submit.
      }
    }, _COLLISION_DEBOUNCE_MS);
  });

  // Submit handler. The collision-flag on the submit button decides
  // whether we send on_collision=overwrite up-front or let the server
  // 409 us back into the warning state.
  const onSubmit = async () => {
    const title = (titleInput?.value || '').trim();
    if (!title) { titleInput?.focus(); return; }
    const description = (descInput?.value || '').trim();
    const onCollision = submitBtn?.dataset.collision === '1' ? 'overwrite' : 'abort';
    submitBtn.disabled = true;
    submitBtn.textContent = onCollision === 'overwrite' ? 'Overwriting…' : 'Saving…';
    try {
      const result = await _save({ workspaceId, title, description, onCollision });
      _closePrompt();
      _toastSuccess(result);
    } catch (exc) {
      submitBtn.disabled = false;
      submitBtn.textContent = onCollision === 'overwrite' ? 'Overwrite →' : 'Save';
      if (exc.status === 409 && exc.body?.error === 'title_collision') {
        _showCollisionWarning(promptEl, exc.body);
      } else if (exc.status === 409 && exc.body?.error === 'dynamic_preview') {
        showToast(exc.body.hint || 'Dynamic preview — save not supported in v1', 'error', 6000);
        _closePrompt();
      } else if (exc.status === 413) {
        showToast(exc.body?.hint || 'Save exceeds storage budget', 'error', 6000);
      } else {
        showToast(exc.message || 'Save failed', 'error');
      }
    }
  };
  submitBtn?.addEventListener('click', onSubmit);
  titleInput?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      onSubmit();
    }
  });

  // Focus the title input on open, with a default suggestion if the
  // preflight gave one.
  if (snap.suggested_title && titleInput) {
    titleInput.value = snap.suggested_title;
    titleInput.select();
  }
  titleInput?.focus();
}

function _toastSuccess(result) {
  // Two-channel feedback: a short success toast for accessibility and
  // the toast region, plus a Play-action banner anchored in the preview
  // pane chrome so the user has an explicit next step without hunting
  // for the saved item.
  const verb = result.action === 'overwritten' ? 'Overwritten' : 'Saved';
  showToast(`${verb}: ${result.title || 'publication'}`, 'success', 4000);
  _showActionBanner(result);
}

function _showActionBanner(result) {
  // Use the same prompt container as the save UI — it's already
  // positioned next to the save button, so the banner appears where
  // the user just clicked. Removed after ~8s or on click anywhere.
  const promptEl = document.getElementById('coder-save-prompt');
  if (!promptEl) return;

  const playUrl = result.launch_url || `/api/library/play/${result.publication_id}`;
  promptEl.classList.remove('hidden');
  promptEl.innerHTML = `
    <div class="coder-save-banner">
      <div class="coder-save-banner-icon">✓</div>
      <div class="coder-save-banner-body">
        <div class="coder-save-banner-title">${escapeHtml(result.title || 'Saved')}</div>
        <div class="coder-save-banner-sub">${result.action === 'overwritten' ? 'Replaced existing save' : 'Added to your Library'}</div>
      </div>
      <a class="btn-primary small" id="coder-save-banner-play" href="${escapeHtml(playUrl)}" target="_blank" rel="noopener">Play ↗</a>
      <button class="coder-save-banner-close" id="coder-save-banner-close" aria-label="Dismiss">✕</button>
    </div>
  `;

  const close = () => {
    promptEl.classList.add('hidden');
    promptEl.innerHTML = '';
  };
  promptEl.querySelector('#coder-save-banner-close')?.addEventListener('click', close);
  // Auto-dismiss after 8s. Play link click does NOT close — the user may
  // want to verify in a tab and then dismiss manually.
  setTimeout(close, 8000);
}

/**
 * Close handle exposed so callers (e.g. coder.js workspace switch)
 * can dismiss the prompt without poking module internals.
 */
export function closeSavePrompt() {
  _closePrompt();
}
