/* ==========================================================================
   Chat Module — ChatInput
   Per-surface input area: textarea, send/stop button, attachment strip
   ========================================================================== */

import { escapeHtml } from '../app.js';
import { ChatAutocomplete } from './autocomplete.js';
import { icons } from './constants.js';
import { scheduleAutosize } from '../utils/textarea-autosize.js';
import { buildSurfaceToolbar } from './toolbar/surface-toolbar.js';

const SEND_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="20" height="20">
  <line x1="22" y1="2" x2="11" y2="13"/>
  <polygon points="22 2 15 22 11 13 2 9 22 2"/>
</svg>`;

const STOP_SVG = `<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
  <rect x="6" y="6" width="12" height="12" rx="2"/>
</svg>`;

export class ChatInput {
  constructor(options = {}) {
    // Surface that owns this composer. Null on the legacy/primary singleton
    // path. Used by Steps 2-6 of the surface-owned composer migration to
    // route toolbar state reads through the focused surface instead of
    // process globals. See docs/superpowers/specs/2026-05-31-surface-owned-composer-design.md.
    this.surface = options.surface || null;
    if (typeof window !== 'undefined' && window.__augmentumDebugComposer) {
      console.debug('[chat-input] constructed with surface=', this.surface?.id ?? '<primary/legacy>');
    }
    this.containerEl = null;
    this.textareaEl = null;
    this.sendBtnEl = null;
    this.attachStripEl = null;
    this._pendingImages = [];
    this._pendingDocs = [];
    this._isStreaming = false;
    this._onSend = options.onSend || (() => {});
    this._onStop = options.onStop || (() => {});
  }

  /* ------------------------------------------------------------------
     DOM creation
     ------------------------------------------------------------------ */

  createDOM(container) {
    const area = document.createElement('div');
    area.className = 'input-area';

    // Attachment preview strip
    const strip = document.createElement('div');
    strip.className = 'input-attachments hidden';
    area.appendChild(strip);
    this.attachStripEl = strip;

    // Per-surface composer toolbar (secondary tabs only). The primary keeps
    // the singleton #input-toolbar it adopts from index.html; independent
    // tabs used to ship a bare textarea+send — the composer gap documented
    // in the surface-owned composer spec §F. buildSurfaceToolbar clones the
    // singleton's markup, strips the not-yet-per-surface controls, and
    // gates visibility by THIS surface's mode.
    if (this.surface && !this.surface._isPrimary) {
      try {
        this._toolbar = buildSurfaceToolbar(this.surface, this, area);
      } catch (err) {
        console.warn('[chat-input] surface toolbar build failed', err);
        this._toolbar = null;
      }
    }

    // Input wrapper (matches existing CSS for rounded border, focus glow, padding)
    const row = document.createElement('div');
    row.className = 'input-wrapper';

    const textarea = document.createElement('textarea');
    textarea.className = 'chat-input';
    textarea.placeholder = 'Type a message...';
    textarea.rows = 1;
    textarea.spellcheck = true;
    row.appendChild(textarea);
    this.textareaEl = textarea;

    const sendBtn = document.createElement('button');
    sendBtn.className = 'send-btn';
    sendBtn.title = 'Send';
    sendBtn.innerHTML = SEND_SVG;
    row.appendChild(sendBtn);
    this.sendBtnEl = sendBtn;

    area.appendChild(row);
    container.appendChild(area);
    this.containerEl = area;

    this._wireEvents();

    // Observation Substrate autocomplete. Self-contained module —
    // fetches suggestions on debounced typing, renders a Tab-to-accept
    // chip below the composer. Silently no-ops if the substrate is off
    // server-side, so it's safe to attach unconditionally.
    try {
      this._autocomplete = new ChatAutocomplete(this.textareaEl, {
        getSurface: () => 'chat',
        getMode: () => (
          // Best-effort current-mode read; the autocomplete query is
          // surface+mode scoped per the L0 fingerprint. Empty when
          // unresolvable (matches the seeder's tolerance for empty mode).
          window.app?.state?.activeSession?.mode ?? ''
        ),
      });
    } catch (err) {
      // Never break the composer over autocomplete wiring.
      console.debug('[chat-input] autocomplete attach failed', err);
    }
  }

  /* ------------------------------------------------------------------
     Event wiring
     ------------------------------------------------------------------ */

  _wireEvents() {
    // Auto-resize on input + pre-warm KV cache while typing
    this._prewarmTimer = null;
    this.textareaEl.addEventListener('input', () => {
      this._autoResize();
      this._schedulePrewarm();
    });

    // Keyboard shortcuts
    this.textareaEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (this._isStreaming) {
          this._onStop();
        } else {
          this._handleSend();
        }
      }
      if (e.key === 'Escape') {
        this.textareaEl.blur();
      }
    });

    // Send / stop button
    this.sendBtnEl.addEventListener('click', () => {
      if (this._isStreaming) {
        this._onStop();
      } else {
        this._handleSend();
      }
    });

    // Paste image from clipboard
    this.textareaEl.addEventListener('paste', (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          e.preventDefault();
          const file = item.getAsFile();
          if (!file) continue;
          const reader = new FileReader();
          reader.onload = () => {
            this.addImage(reader.result);
          };
          reader.readAsDataURL(file);
        }
      }
    });
  }

  /* ------------------------------------------------------------------
     Internal helpers
     ------------------------------------------------------------------ */

  _handleSend() {
    const text = this.textareaEl.value.trim();
    const images = [...this._pendingImages];
    const docs = [...this._pendingDocs];
    if (!text && images.length === 0 && docs.length === 0) return;
    this._onSend(text, images, docs);
    this.clear();
  }

  _autoResize() {
    // rAF-deferred + cached-height short-circuit. The naive pattern
    // (height='auto' + read scrollHeight + write height) forces
    // synchronous layout on every keystroke and shows up as 30-150ms
    // INP per character with streaming content active. See
    // ``ui/scripts/utils/textarea-autosize.js`` for full rationale.
    scheduleAutosize(this.textareaEl);
  }

  _schedulePrewarm() {
    // Debounce: fire 2s after user stops typing to pre-warm the KV cache.
    // By the time they press Enter, the prefix is already processed.
    clearTimeout(this._prewarmTimer);
    const text = this.textareaEl?.value?.trim();
    if (!text || text.length < 5 || this._isStreaming) return;

    this._prewarmTimer = setTimeout(() => {
      this._doPrewarm();
    }, 2000);
  }

  async _doPrewarm() {
    // Draft-aware speculation (KV ladder rung 3). The server rebuilds
    // the exact augmented prefix from its replay ledger and either
    // warms it or fully pre-generates the turn — the old bare-messages
    // /v2/prewarm body never byte-matched injected modes, so it warmed
    // the wrong prefix. Server gates everything (local engine only,
    // idle GPU only, setting default-off); this is fire-and-forget.
    try {
      const session = window.app?.state?.activeSession;
      if (!session?.id || !session?.tree) return;
      const draft = this.textareaEl?.value ?? '';
      if (!draft.trim()) return;

      // The replay ledger row is captured before the reply exists, so
      // the server needs the last assistant message on the active
      // branch to complete its predicted prefix.
      const { getMessagesForLLM } = await import('./tree.js');
      const messages = getMessagesForLLM(session);
      const lastAssistant = [...messages].reverse().find((m) => m?.role === 'assistant');
      const priorAssistant =
        typeof lastAssistant?.content === 'string' ? lastAssistant.content : '';

      fetch('/api/engine/v2/kv/speculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.id,
          draft,
          prior_assistant: priorAssistant,
        }),
      }).catch(() => {}); // silent fail is fine
    } catch {
      // Speculation is best-effort — never disrupt typing
    }
  }

  /* ------------------------------------------------------------------
     Public methods
     ------------------------------------------------------------------ */

  focus() {
    this.textareaEl?.focus();
  }

  /** Re-gate the per-surface toolbar when the owning surface's mode flips
   *  in place (left-panel session click switching passthrough↔analytical). */
  updateToolbarMode(mode) {
    this._toolbar?.updateMode?.(mode);
  }

  getValue() {
    return this.textareaEl?.value ?? '';
  }

  setValue(text) {
    if (!this.textareaEl) return;
    this.textareaEl.value = text;
    this._autoResize();
  }

  clear() {
    if (!this.textareaEl) return;
    this.textareaEl.value = '';
    this.textareaEl.style.height = 'auto';
    this._pendingImages = [];
    this._pendingDocs = [];
    this._updateAttachmentStrip();
  }

  insertText(text) {
    if (!this.textareaEl) return;
    const ta = this.textareaEl;
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    const before = ta.value.substring(0, start);
    const after = ta.value.substring(end);
    ta.value = before + text + after;
    const cursor = start + text.length;
    ta.selectionStart = cursor;
    ta.selectionEnd = cursor;
    this._autoResize();
    ta.focus();
  }

  setStreaming(isStreaming) {
    this._isStreaming = isStreaming;
    if (!this.sendBtnEl) return;
    if (isStreaming) {
      this.sendBtnEl.innerHTML = STOP_SVG;
      this.sendBtnEl.title = 'Stop';
    } else {
      this.sendBtnEl.innerHTML = SEND_SVG;
      this.sendBtnEl.title = 'Send';
    }
  }

  addImage(dataUrl) {
    this._pendingImages.push(dataUrl);
    this._updateAttachmentStrip();
  }

  removeImage(index) {
    this._pendingImages.splice(index, 1);
    this._updateAttachmentStrip();
  }

  addDocument(doc) {
    this._pendingDocs.push(doc);
    this._updateAttachmentStrip();
  }

  getPendingImages() {
    return [...this._pendingImages];
  }

  getPendingDocs() {
    return [...this._pendingDocs];
  }

  /* ------------------------------------------------------------------
     Attachment strip rendering
     ------------------------------------------------------------------ */

  _updateAttachmentStrip() {
    if (!this.attachStripEl) return;

    const hasAttachments = this._pendingImages.length > 0 || this._pendingDocs.length > 0;
    this.attachStripEl.classList.toggle('hidden', !hasAttachments);

    if (!hasAttachments) {
      this.attachStripEl.innerHTML = '';
      return;
    }

    const parts = [];

    for (let i = 0; i < this._pendingImages.length; i++) {
      parts.push(
        `<div class="input-attachment">` +
          `<img src="${this._pendingImages[i]}" alt="Attached" class="input-attachment-thumb" />` +
          `<button class="input-attachment-remove" data-idx="${i}">&times;</button>` +
        `</div>`
      );
    }

    for (let i = 0; i < this._pendingDocs.length; i++) {
      parts.push(
        `<div class="input-attachment input-attachment-doc">` +
          `<span>${escapeHtml(this._pendingDocs[i].filename)}</span>` +
          `<button class="input-attachment-remove" data-doc-idx="${i}">&times;</button>` +
        `</div>`
      );
    }

    this.attachStripEl.innerHTML = parts.join('');

    // Wire remove buttons for images
    this.attachStripEl.querySelectorAll('[data-idx]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.currentTarget.dataset.idx, 10);
        this.removeImage(idx);
      });
    });

    // Wire remove buttons for docs
    this.attachStripEl.querySelectorAll('[data-doc-idx]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.currentTarget.dataset.docIdx, 10);
        this._pendingDocs.splice(idx, 1);
        this._updateAttachmentStrip();
      });
    });
  }

  /* ------------------------------------------------------------------
     Cleanup
     ------------------------------------------------------------------ */

  destroy() {
    // Tear down the per-surface toolbar first: it owns body-mounted
    // artifacts (tools dropdown) and window listeners that would outlive
    // the container removal below.
    try { this._toolbar?.cleanup?.(); } catch { /* best-effort */ }
    this._toolbar = null;
    if (this.containerEl?.parentNode) {
      this.containerEl.parentNode.removeChild(this.containerEl);
    }
    this.containerEl = null;
    this.textareaEl = null;
    this.sendBtnEl = null;
    this.attachStripEl = null;
    this._pendingImages = [];
    this._pendingDocs = [];
    this._onSend = null;
    this._onStop = null;
  }
}
