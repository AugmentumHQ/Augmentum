/**
 * Drag-and-drop attachment handling for the coder conversation pane.
 *
 * Two-path design:
 *
 *   IMAGES → ``/api/chat-images`` (existing chat_images pipeline, already
 *            used by Chat / Narrative modes). The returned URL lives on
 *            ``msg.images`` and is resolved to a base64 data URL server-
 *            side (``augmentum/models/base.py:resolve_chat_image_urls``)
 *            before the VL model sees it. Qwen-VL / Gemma-VL / etc.
 *            receive the image inline as multimodal content.
 *
 *   FILES  → ``/api/coder/files/<workspaceId>/upload`` (existing
 *            workspace upload). Dropped into ``/workspace/.augmentum/
 *            attachments/`` so they don't pollute the user's tree. The
 *            outgoing message text gains a "📎 Attached: <path>" footer
 *            so the agent can ``file_read`` the content on demand.
 *
 * Both flows converge in :func:`buildMessagePayload`, which produces the
 * ``{content, images}`` pair we push into the coder chat history and
 * ultimately send through to ``/api/chat``. Non-image attachments also
 * generate a text reference so models that can't see images still have
 * a path to work with.
 *
 * Scope — Sprint 1:
 *   • Drop onto #coder-conversation (chat pane) only; drops on the file
 *     tree keep their existing workspace-upload behaviour (see
 *     _initUploadAndDrop in coder.js).
 *   • One chip per attachment, inline between the textarea and send
 *     button. Click X to remove before send.
 *   • No progress bars on uploads. For MVP this is acceptable — images
 *     under 20MB upload in well under a second on localhost; the size
 *     cap upstream returns 413 with a clean toast.
 *   • No paste-from-clipboard for images. That's the obvious next win
 *     but requires its own ``paste`` handler.
 *
 * Deferred:
 *   • Multi-image grid preview (lightbox on chip click).
 *   • Paste-from-clipboard (Ctrl+V an image directly into the composer).
 *   • Non-image file thumbnails (PDF first page, text file first-line
 *     preview).
 */
import { escapeHtml, showToast } from './app.js';


// ---------------------------------------------------------------------------
// Classification + upload
// ---------------------------------------------------------------------------

const _IMAGE_MIME_RE = /^image\/(png|jpe?g|webp|gif|bmp|avif|svg\+xml)$/i;


/**
 * Read a File as a base64 data URL. Promise wrapper around FileReader
 * so the caller can ``await`` it cleanly; FileReader's event-driven API
 * makes async code read awkwardly otherwise.
 */
function _readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = () => reject(new Error('read failed'));
    fr.readAsDataURL(file);
  });
}


/**
 * POST with upload progress. fetch() doesn't expose upload progress
 * without ReadableStream gymnastics; XHR does via
 * ``xhr.upload.addEventListener('progress')``. Returns a Promise
 * that resolves to the parsed JSON response body, or rejects with an
 * Error carrying the server's ``error`` field when present.
 *
 * @param {string}              url
 * @param {FormData|BodyInit}   body
 * @param {(loaded:number, total:number) => void} onProgress
 * @param {object}              [headers]
 */
function _xhrUploadJson(url, body, onProgress, headers = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    for (const [k, v] of Object.entries(headers)) {
      // Skip Content-Type for FormData — browser must set the
      // multipart boundary automatically.
      if (k.toLowerCase() === 'content-type' && (body instanceof FormData)) continue;
      xhr.setRequestHeader(k, v);
    }
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total);
    });
    xhr.addEventListener('load', () => {
      let parsed = null;
      try { parsed = JSON.parse(xhr.responseText); } catch { parsed = null; }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parsed || {});
      } else {
        const msg = parsed?.error || `HTTP ${xhr.status}`;
        reject(new Error(msg));
      }
    });
    xhr.addEventListener('error', () => reject(new Error('network error')));
    xhr.addEventListener('abort',  () => reject(new Error('upload aborted')));
    xhr.send(body);
  });
}


/**
 * Upload an image file as a chat_images row and return the stable URL.
 * Returns null on failure (network, size cap hit, server error).
 *
 * @param {File}   file       — the image
 * @param {string} sessionId  — workspace/session id for chat_images scoping
 * @param {(pct:number) => void} [onProgress]
 *                            — called with 0..100 during encode + upload
 */
export async function uploadImageAsChat(file, sessionId = '', onProgress = null) {
  try {
    // Stage 1: base64-encode client-side (FileReader has no progress
    // for files under a few MB; we just flip to 50% when it's done,
    // which is accurate enough for the chip UI).
    const dataUrl = await _readFileAsDataUrl(file);
    onProgress?.(50);

    // Stage 2: POST the JSON body. XHR for upload progress; the
    // progress event fires on the REQUEST body, so we map bytes
    // uploaded → percentage of the remaining 50% slice.
    const body = JSON.stringify({
      data_url:   dataUrl,
      session_id: sessionId,
    });
    const data = await _xhrUploadJson(
      '/api/chat-images', body,
      (loaded, total) => {
        if (!total) return;
        // Map upload bytes into the 50-100 range.
        const pct = 50 + Math.round((loaded / total) * 50);
        onProgress?.(Math.min(pct, 99));
      },
      { 'Content-Type': 'application/json' },
    );
    onProgress?.(100);
    return data.url || null;  // e.g. /api/chat-images/abc123
  } catch (err) {
    showToast('Image upload failed: ' + err.message, 'error', 5000);
    return null;
  }
}


/**
 * Upload a non-image file into the workspace's attachments folder.
 * Returns the absolute container path on success, null on failure.
 *
 * @param {string} workspaceId
 * @param {File}   file
 * @param {(pct:number) => void} [onProgress]
 */
export async function uploadFileToWorkspace(workspaceId, file, onProgress = null) {
  if (!workspaceId) return null;
  try {
    const form = new FormData();
    form.append('dest_path', '/workspace/.augmentum/attachments');
    form.append('files', file, file.name);

    await _xhrUploadJson(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/upload`,
      form,
      (loaded, total) => {
        if (!total) return;
        const pct = Math.round((loaded / total) * 100);
        onProgress?.(Math.min(pct, 99));
      },
    );
    onProgress?.(100);
    // The backend returns `uploaded: N` but not the paths. We construct
    // the path ourselves — the upload route writes each file under
    // dest_path using its original filename. If the backend ever
    // renames (collision handling), a future enhancement here would
    // parse an echoed path list. For MVP the reconstruction is
    // accurate.
    return `/workspace/.augmentum/attachments/${file.name}`;
  } catch (err) {
    showToast('Upload failed: ' + err.message, 'error', 5000);
    return null;
  }
}


/**
 * Classify + upload a single dropped file. Returns an attachment
 * descriptor or null if upload failed.
 *
 * Shape of the returned descriptor:
 *   {
 *     id:    string,   // client-side only — for chip DOM keying
 *     kind:  'image' | 'file',
 *     name:  string,   // original filename for display
 *     size:  number,   // bytes
 *     mime:  string,
 *     url:   string,   // images: chat-images URL; files: workspace path
 *     dataUrl: string | null,  // images only, for thumbnail preview
 *   }
 *
 * @param {File}   file
 * @param {string} workspaceId
 * @param {string} sessionId
 */
export async function ingestFile(file, workspaceId, sessionId, onProgress = null) {
  const isImage = _IMAGE_MIME_RE.test(file.type || '');

  let url;
  let dataUrl = null;

  if (isImage) {
    // Read + keep the data URL client-side for a no-round-trip chip
    // thumbnail. Uploading BELOW reads the file again into a fresh
    // FileReader — one extra read is fine, and it means we can
    // render the chip before the upload completes.
    try { dataUrl = await _readFileAsDataUrl(file); } catch { /* fall through */ }
    url = await uploadImageAsChat(file, sessionId, onProgress);
  } else {
    url = await uploadFileToWorkspace(workspaceId, file, onProgress);
  }

  if (!url) return null;

  return {
    id:      `att_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    kind:    isImage ? 'image' : 'file',
    name:    file.name,
    size:    file.size,
    mime:    file.type || '',
    url,
    dataUrl,
  };
}


/**
 * Build a placeholder descriptor for optimistic chip rendering BEFORE
 * the upload completes. The caller inserts a chip via :func:`renderChip`,
 * then calls :func:`ingestFile` with an ``onProgress`` callback that
 * updates chip state. When upload resolves, the caller swaps the
 * placeholder's ``url`` / ``id`` with the real descriptor returned
 * by ``ingestFile``.
 *
 * @param {File} file
 */
export async function buildPendingDescriptor(file) {
  const isImage = _IMAGE_MIME_RE.test(file.type || '');
  let dataUrl = null;
  if (isImage) {
    try { dataUrl = await _readFileAsDataUrl(file); } catch { /* best effort */ }
  }
  return {
    id:      `pending_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    kind:    isImage ? 'image' : 'file',
    name:    file.name,
    size:    file.size,
    mime:    file.type || '',
    url:     null,    // filled in when upload completes
    dataUrl,
    pending: true,    // chip renders progress bar while truthy
  };
}


// ---------------------------------------------------------------------------
// Rendering — chip DOM
// ---------------------------------------------------------------------------

/**
 * Render a chip for an attachment. Returns the root element so callers
 * can append + wire events. The remove-X callback is owned by the
 * caller so the chip doesn't need to know about the attachment array.
 *
 * @param {object}   attachment  — descriptor from ingestFile()
 * @param {function} onRemove    — called when user clicks the X
 */
export function renderChip(attachment, onRemove) {
  const chip = document.createElement('div');
  chip.className = `coder-attachment-chip coder-attachment-chip--${attachment.kind}`;
  if (attachment.pending) chip.classList.add('is-pending');
  chip.dataset.attachmentId = attachment.id;

  const body = attachment.kind === 'image' && attachment.dataUrl
    ? `<img class="coder-attachment-thumb" src="${escapeHtml(attachment.dataUrl)}" alt="">`
    : `<span class="coder-attachment-icon" aria-hidden="true">${_iconForMime(attachment.mime, attachment.name)}</span>`;

  chip.innerHTML = `
    ${body}
    <span class="coder-attachment-meta">
      <span class="coder-attachment-name" title="${escapeHtml(attachment.name)}">
        ${escapeHtml(attachment.name)}
      </span>
      <span class="coder-attachment-size">${_fmtBytes(attachment.size)}</span>
      <span class="coder-attachment-progress" role="progressbar"
            aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
        <span class="coder-attachment-progress-fill"></span>
      </span>
    </span>
    <button type="button" class="coder-attachment-remove" title="Remove" aria-label="Remove attachment">&times;</button>
  `;

  chip.querySelector('.coder-attachment-remove')?.addEventListener('click', (e) => {
    e.stopPropagation();
    onRemove?.(attachment);
  });

  return chip;
}


/**
 * Update an existing chip's progress bar + completion state. Called
 * from the ``onProgress`` callback passed to :func:`ingestFile` and
 * from the post-upload commit path.
 *
 * @param {HTMLElement} chipEl
 * @param {number}      pct       — 0..100; ``null`` signals "completed,
 *                                   switch to ready state"
 */
export function updateChipProgress(chipEl, pct) {
  if (!chipEl) return;
  const bar = chipEl.querySelector('.coder-attachment-progress');
  const fill = chipEl.querySelector('.coder-attachment-progress-fill');
  if (pct === null || pct >= 100) {
    chipEl.classList.remove('is-pending');
    if (bar) bar.style.display = 'none';
    return;
  }
  chipEl.classList.add('is-pending');
  if (bar) {
    bar.style.display = '';
    bar.setAttribute('aria-valuenow', String(Math.round(pct)));
  }
  if (fill) fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}


/**
 * Mark a chip as failed — caller typically removes after a short
 * delay so the user sees the error state flash. Used from
 * upload-failure paths where the descriptor would otherwise have
 * been silently removed.
 */
export function markChipFailed(chipEl) {
  if (!chipEl) return;
  chipEl.classList.remove('is-pending');
  chipEl.classList.add('is-failed');
}


// ---------------------------------------------------------------------------
// Payload construction
// ---------------------------------------------------------------------------

/**
 * Build the outgoing chat message payload from composer text +
 * attachments.
 *
 * For IMAGES: URLs land on ``images[]``. The backend's
 * ``resolve_chat_image_urls`` expands them to base64 data URLs the VL
 * model can see directly. No text reference is added — the model
 * literally sees the image, so a path is noise.
 *
 * For FILES: paths are appended to the text as "📎 Attached: <path>"
 * lines. The model then has a concrete path to ``file_read`` on its
 * next turn. Multiple files produce multiple lines. When the user
 * supplied no text at all, a minimal prompt ("Please take a look at
 * the attached file(s).") is inserted so the model has SOMETHING to
 * act on — otherwise a pure-attachment message would read as empty to
 * the plan phase and route to the vague-request path.
 *
 * Returns ``{content, images}``. ``images`` is ``undefined`` when
 * there are no image attachments (avoids sending an empty array that
 * could trip older backends).
 *
 * @param {string}   text
 * @param {object[]} attachments
 */
export function buildMessagePayload(text, attachments) {
  attachments = attachments || [];

  const images = attachments
    .filter(a => a.kind === 'image' && a.url)
    .map(a => a.url);

  const files = attachments.filter(a => a.kind === 'file' && a.url);

  let content = (text || '').trim();

  if (files.length) {
    const lines = files.map(f => `📎 Attached: ${f.url}`);
    if (content) {
      content = `${content}\n\n${lines.join('\n')}`;
    } else {
      content = [
        'Please take a look at the attached file(s).',
        ...lines,
      ].join('\n');
    }
  } else if (!content && images.length) {
    // Pure image drop with no text. VL models handle "empty prompt +
    // image" via the ``_VISION_DEFAULT_PROMPT`` injection in models/
    // base.py, but plan phase would still see empty text and might
    // route to vague-request. Insert a tiny seed so the plan phase
    // recognises the turn as a real request.
    content = 'What do you see?';
  }

  const out = { content };
  if (images.length) out.images = images;
  return out;
}


// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function _fmtBytes(n) {
  n = Number(n) || 0;
  if (n < 1024)             return `${n} B`;
  if (n < 1024 * 1024)      return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(1)} GB`;
}


function _iconForMime(mime, name) {
  // Lean on file extension when MIME is missing (drag from OS may not
  // always populate type). Simple symbolic icons — no emoji since
  // they're semantically ambiguous across themes.
  const lower = (mime || '').toLowerCase();
  const ext = (name || '').toLowerCase().split('.').pop() || '';

  if (lower.startsWith('text/') || ['txt', 'md', 'log', 'csv'].includes(ext)) return '≡';
  if (lower === 'application/pdf' || ext === 'pdf') return 'PDF';
  if (lower === 'application/json' || ext === 'json') return '{}';
  if (['zip', 'tar', 'gz', '7z', 'rar'].includes(ext) || lower.includes('zip')) return '□';
  if (['js', 'ts', 'py', 'rs', 'go', 'c', 'cpp', 'java'].includes(ext)) return '</>';
  return '▢';
}
