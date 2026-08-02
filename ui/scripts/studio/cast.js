/* ==========================================================================
   Studio cast — send an artifact's preview to a paired TV.
   --------------------------------------------------------------------------
   Self-contained: lists receivers via /api/cast/receivers, shows a picker
   modal, dispatches via /api/cast/send with surface_kind='html.generic' and
   the artifact preview URL. Studio owns its own cast flow so this module
   stays decoupled from games / library cast paths.

   Public:
     castArtifactPreview(artifactId, displayName) → Promise<{...} | reject>

   Throws Error('cancelled') when the user dismisses the picker. Real
   failures (network, no receivers) surface their own toasts and re-throw.
   ========================================================================== */

import { escapeHtml, showToast } from '../app.js';

/**
 * Open the receiver picker and cast the artifact's preview to the chosen TV.
 *
 * @param {string} artifactId — the artifact's UUID
 * @param {string} displayName — shown on the receiver during cast
 * @returns {Promise<object>} dispatch response augmented with receiver info
 */
export async function castArtifactPreview(artifactId, displayName) {
  if (!artifactId) throw new Error('castArtifactPreview requires an artifactId');

  let receivers;
  try {
    receivers = await _listReceivers();
  } catch (err) {
    showToast(`Couldn’t list TVs: ${err.message || err}`, 'error');
    throw err;
  }

  const receiverId = await _pickReceiver(receivers);

  const surfaceUrl = `/api/artifacts/${encodeURIComponent(artifactId)}/preview`;
  let resp;
  try {
    resp = await _dispatch(receiverId, artifactId, displayName, surfaceUrl);
  } catch (err) {
    showToast(`Cast failed: ${err.message || err}`, 'error');
    throw err;
  }

  const receiverLabel = receivers.find(r => r.registration_id === receiverId)?.label || 'your TV';
  showToast(`Now showing on ${receiverLabel}.`, 'info');
  return { ...resp, receiver_id: receiverId, receiver_label: receiverLabel };
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

async function _listReceivers() {
  const r = await fetch('/api/cast/receivers', { credentials: 'same-origin' });
  if (!r.ok) throw new Error(`receivers HTTP ${r.status}`);
  const body = await r.json();
  return Array.isArray(body.receivers) ? body.receivers : [];
}

async function _dispatch(receiverId, artifactId, displayName, surfaceUrl) {
  const r = await fetch('/api/cast/send', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({
      receiver_id: receiverId,
      surface_kind: 'html.generic',
      surface_url: surfaceUrl,
      slot: 'main',
      state: {
        title: displayName || 'Artifact',
        cast_source: 'studio',
        artifact_id: artifactId,
        cast_input_config: null,
        cast_strategy: 'shim',
      },
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `HTTP ${r.status}`);
  }
  return r.json();
}

function _pickReceiver(receivers) {
  document.querySelectorAll('.studio-cast-picker').forEach(el => el.remove());

  return new Promise((resolve, reject) => {
    const overlay = document.createElement('div');
    overlay.className = 'studio-cast-picker';
    const card = document.createElement('div');
    card.style.cssText = `
      background: #1c1c1f; border: 1px solid #2c2c2c; border-radius: 8px;
      max-width: 420px; width: 100%; padding: 20px; color: #e7e9ee;
      font: 13px/1.4 "Source Sans 3", "Inter", system-ui, sans-serif;
    `;

    const head = document.createElement('div');
    head.style.cssText = 'font-size: 15px; font-weight: 600; margin-bottom: 8px;';
    head.textContent = receivers.length ? 'Cast to TV' : 'No TVs connected';
    card.appendChild(head);

    const sub = document.createElement('div');
    sub.style.cssText = 'font-size: 12px; opacity: 0.6; margin-bottom: 16px;';
    sub.textContent = receivers.length
      ? 'Pick a receiver — the preview opens immediately.'
      : 'Open Augmentum on your TV and pair it first.';
    card.appendChild(sub);

    if (receivers.length) {
      const list = document.createElement('div');
      list.style.cssText = 'display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px;';
      for (const r of receivers) {
        const row = document.createElement('button');
        row.type = 'button';
        row.style.cssText = `
          background: #232328; border: 1px solid #2c2c2c; border-radius: 6px;
          padding: 10px 12px; color: #e7e9ee; text-align: left; cursor: pointer;
          display: flex; align-items: center; gap: 10px;
        `;
        row.innerHTML = `
          <span style="font-size: 16px;">📺</span>
          <span style="flex: 1;">
            <div style="font-weight: 500;">${escapeHtml(r.label || 'Untitled TV')}</div>
            <div style="font-size: 11px; opacity: 0.55;">${escapeHtml(r.platform || 'receiver')}</div>
          </span>
        `;
        row.addEventListener('mouseenter', () => row.style.borderColor = '#6ea2ef');
        row.addEventListener('mouseleave', () => row.style.borderColor = '#2c2c2c');
        row.addEventListener('click', () => {
          overlay.remove();
          document.removeEventListener('keydown', esc);
          resolve(r.registration_id);
        });
        list.appendChild(row);
      }
      card.appendChild(list);
    }

    const actions = document.createElement('div');
    actions.style.cssText = 'display: flex; justify-content: flex-end; gap: 8px;';
    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.cssText = `
      background: transparent; border: 1px solid #333; color: #ccc;
      padding: 6px 14px; border-radius: 4px; cursor: pointer;
    `;
    cancelBtn.addEventListener('click', () => {
      overlay.remove();
      document.removeEventListener('keydown', esc);
      reject(new Error('cancelled'));
    });
    actions.appendChild(cancelBtn);
    card.appendChild(actions);

    overlay.appendChild(card);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        overlay.remove();
        document.removeEventListener('keydown', esc);
        reject(new Error('cancelled'));
      }
    });

    const esc = (e) => {
      if (e.key === 'Escape') {
        document.removeEventListener('keydown', esc);
        overlay.remove();
        reject(new Error('cancelled'));
      }
    };
    document.addEventListener('keydown', esc);

    document.body.appendChild(overlay);
  });
}
