/* ==========================================================================
   companion-context.js — the "Read this …" handoff.

   The perception bridge (architect-observer.js) auto-reports index/digest
   fidelity on every focus change — a page title, a file name. This module
   is the opt-in DEEP channel: a surface registers a loadable provider when
   it has full content the user might want the companion to actually read,
   and the companion widget shows a context-aware button. Pressing it pulls
   the provider's full content and POSTs it to /api/architect/load_context,
   where it lands in the LoadedContextStore for context_peek('loaded').

   One provider at a time — the foreground surface owns it. Clearing only
   fires for the surface that currently owns the slot, so a backgrounded
   surface's blur can't wipe the foreground's provider.

   Labels are persona-agnostic ("Read this page", never "Show her …") —
   the UI never names the companion; deployments configure their own.
   ========================================================================== */

const LOAD_URL = '/api/architect/load_context';

// Per-kind button copy. Falls back to "Read this <kind>" for unknown kinds.
const _KIND_LABEL = {
  page:  'Read this page',
  chat:  'Catch up on this chat',
  file:  'Read this file',
  scene: 'Catch up on this scene',
  book:  'Read this book',
};

// { kind, label, getContent } — getContent may be sync or async and returns
// { label?, content, ref? }. Null when nothing is loadable.
let _active = null;

/**
 * Register the foreground surface's loadable content. ``getContent`` is
 * called lazily (only when the user presses the button), so a surface can
 * register cheaply on focus and do the expensive extraction on demand.
 */
export function setCompanionLoadable(kind, label, getContent) {
  if (!kind || typeof getContent !== 'function') return;
  _active = { kind: String(kind), label: label || '', getContent };
  _broadcast();
}

/** Clear the slot — only if the clearing surface currently owns it. */
export function clearCompanionLoadable(kind) {
  if (_active && (!kind || _active.kind === kind)) {
    _active = null;
    _broadcast();
  }
}

export function getCompanionLoadable() {
  return _active;
}

/**
 * Convenience: register a provider that reads its content from a visible
 * DOM container's ``innerText`` on demand. For non-iframe surfaces whose
 * transcript/buffer lives in the page (chat, coder). Returns false (and
 * registers nothing) when the selector matches no element or it's empty —
 * so the chip never offers to read something that isn't there.
 */
export function setLoadableFromDom(kind, label, selector) {
  const probe = () => {
    const el = document.querySelector(selector);
    return el ? String(el.innerText || '').trim() : '';
  };
  if (!probe()) {
    clearCompanionLoadable(kind);
    return false;
  }
  setCompanionLoadable(kind, label, () => ({
    label: label || '', content: probe(),
  }));
  return true;
}

/** Button copy for the active provider, or '' when nothing is loadable. */
export function loadableButtonLabel() {
  if (!_active) return '';
  return _KIND_LABEL[_active.kind] || `Read this ${_active.kind}`;
}

function _broadcast() {
  try {
    document.dispatchEvent(new CustomEvent('augmentum:loadable-changed', {
      detail: { active: Boolean(_active), label: loadableButtonLabel() },
    }));
  } catch (_) { /* no document (tests) — silent */ }
}

/**
 * Pull the active provider's full content and hand it to the companion.
 * Returns ``{ok, chars}`` on success or ``{ok:false, reason}``. Never
 * throws — the caller toasts the outcome.
 */
export async function loadCompanionContext() {
  if (!_active) return { ok: false, reason: 'nothing-loadable' };
  let payload;
  try {
    payload = await _active.getContent();
  } catch (err) {
    console.warn('[companion-context] provider threw', err);
    return { ok: false, reason: 'provider-error' };
  }
  const content = String((payload && payload.content) || '');
  if (!content.trim()) return { ok: false, reason: 'empty' };
  try {
    const resp = await fetch(LOAD_URL, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind: _active.kind,
        label: (payload && payload.label) || _active.label || '',
        content,
        ref: (payload && payload.ref) || '',
      }),
    });
    if (!resp.ok) return { ok: false, reason: `http-${resp.status}` };
    const data = await resp.json().catch(() => ({}));
    return { ok: true, chars: data.chars || content.length };
  } catch (err) {
    console.warn('[companion-context] load failed', err);
    return { ok: false, reason: 'network' };
  }
}
