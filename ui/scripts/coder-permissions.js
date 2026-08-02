// Coder permission-approval modal
// ---------------------------------------------------------------------------
// Polls /v1/coder/permissions/pending every few seconds while the coder
// surface is active. For each pending request, shows an approval modal
// with the tool name, input JSON, and Allow / Deny buttons. The POST
// response resolves the backend's awaited asyncio.Future so the coder
// agent can proceed (or see a structured denial).

const POLL_INTERVAL_MS = 2000;
const PENDING_URL = '/v1/coder/permissions/pending';

let _pollTimer = null;
let _modalEl = null;
let _activeRequestId = null;

function _escapeHtml(s) {
  if (typeof s !== 'string') s = String(s ?? '');
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/`/g, '&#96;')
    .replace(/\$\{/g, '&#36;&#123;');
}

function _ensureModal() {
  if (_modalEl) return _modalEl;
  const el = document.createElement('div');
  el.id = 'coder-permission-modal';
  el.className = 'coder-permission-modal hidden';
  el.innerHTML = `
    <div class="coder-permission-backdrop"></div>
    <div class="coder-permission-dialog" role="dialog" aria-label="Tool permission required">
      <div class="coder-permission-header">
        <span class="coder-permission-title">Coder wants to use a tool</span>
      </div>
      <div class="coder-permission-body">
        <div class="coder-permission-row">
          <span class="coder-permission-label">Tool:</span>
          <span class="coder-permission-tool" id="coder-permission-tool"></span>
        </div>
        <div class="coder-permission-row">
          <span class="coder-permission-label">Input:</span>
          <pre class="coder-permission-input" id="coder-permission-input"></pre>
        </div>
      </div>
      <div class="coder-permission-actions">
        <button class="btn btn-secondary" id="coder-permission-deny">Deny</button>
        <button class="btn btn-primary" id="coder-permission-approve">Allow</button>
      </div>
    </div>
  `;
  document.body.appendChild(el);

  el.querySelector('#coder-permission-approve').addEventListener('click', () => _resolve(true));
  el.querySelector('#coder-permission-deny').addEventListener('click', () => _resolve(false));
  el.querySelector('.coder-permission-backdrop').addEventListener('click', () => _resolve(false));

  _modalEl = el;
  return el;
}

function _showModal(req) {
  const el = _ensureModal();
  _activeRequestId = req.id;
  el.querySelector('#coder-permission-tool').textContent = req.tool_name;
  let prettyInput;
  try {
    prettyInput = JSON.stringify(req.tool_input, null, 2);
  } catch {
    prettyInput = String(req.tool_input);
  }
  el.querySelector('#coder-permission-input').textContent = prettyInput;
  el.classList.remove('hidden');
}

function _hideModal() {
  _activeRequestId = null;
  if (_modalEl) _modalEl.classList.add('hidden');
}

async function _resolve(approved) {
  const id = _activeRequestId;
  if (!id) {
    _hideModal();
    return;
  }
  const endpoint = approved ? 'approve' : 'deny';
  try {
    await fetch(`/v1/coder/permissions/${encodeURIComponent(id)}/${endpoint}`, {
      method: 'POST',
    });
  } catch (err) {
    console.warn('coder permission resolve failed', err);
  }
  _hideModal();
  // Poll again immediately so a queued-up request surfaces without the
  // 2-second gap the user just spent reading the modal.
  _poll();
}

async function _poll() {
  // A backgrounded tab can't show the approval modal anyway — skip the
  // fetch (same visibility discipline as the other coder polls). The
  // request re-surfaces within one interval of the tab foregrounding.
  if (document.hidden) return;
  try {
    const resp = await fetch(PENDING_URL);
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.enabled) return;
    const pending = data.pending || [];
    if (pending.length === 0) {
      if (_activeRequestId) _hideModal();
      return;
    }
    // Show the first pending request. Don't stack — one modal at a time.
    const current = pending[0];
    if (current.id !== _activeRequestId) {
      _showModal(current);
    }
  } catch (err) {
    // Network hiccups are fine — just try again next tick.
  }
}

export function startCoderPermissionListener() {
  if (_pollTimer) return;
  _pollTimer = setInterval(_poll, POLL_INTERVAL_MS);
  _poll();  // first tick immediate
}

export function stopCoderPermissionListener() {
  if (_pollTimer) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
  _hideModal();
}
