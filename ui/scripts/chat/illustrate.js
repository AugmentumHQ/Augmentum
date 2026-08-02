/* ==========================================================================
   Chat Module — Illustrate Moment
   Narrative text selection → scene image generation
   ========================================================================== */

import { escapeHtml, extractErrorMessage, showToast } from '../app.js';
import { buildMessagesForAPI } from './tree.js';
import { createImageProgressLoader } from './image-progress.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _illustrateTooltip = null;
let _illustrateSelection = null;

/** Injected getter — returns the active session object. */
let _getSession = () => null;

/** Injected getter — returns the current mode string. */
let _getMode = () => 'passthrough';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Wire the session getter so _doIllustrateMoment can access getActiveSession().
 */
export function setIllustrateSessionGetter(fn) {
  _getSession = fn;
}

/**
 * Wire the mode getter so _onSelectionChange can check narrative mode.
 */
export function setIllustrateModeGetter(fn) {
  _getMode = fn;
}

/**
 * Create the floating tooltip, attach selectionchange + click-outside listeners.
 * Call once during init.
 */
export function initIllustrateMoment() {
  // Create the floating tooltip (created once, repositioned on each selection)
  _illustrateTooltip = document.createElement('div');
  _illustrateTooltip.className = 'illustrate-tooltip hidden';
  _illustrateTooltip.innerHTML = `
    <button class="illustrate-btn">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>
      </svg>
      <span>Illustrate this moment</span>
    </button>
  `;
  document.body.appendChild(_illustrateTooltip);

  // Handle the illustrate button click
  _illustrateTooltip.querySelector('.illustrate-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    if (_illustrateSelection) {
      _doIllustrateMoment(_illustrateSelection);
    }
    _hideIllustrateTooltip();
  });

  // Listen for text selection changes in the chat area
  document.addEventListener('selectionchange', _onSelectionChange);

  // Hide on click outside — but keep it visible while a text selection still
  // exists (otherwise the mouseup/click that ENDED the selection dismisses the
  // tooltip the same frame it was shown).
  document.addEventListener('click', (e) => {
    if (e.target.closest('.illustrate-tooltip')) return;
    const sel = window.getSelection();
    const hasSelection = !!(sel && !sel.isCollapsed && sel.toString().trim().length > 0);
    if (!hasSelection) _hideIllustrateTooltip();
  });
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function _onSelectionChange() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.toString().trim()) {
    // Small delay before hiding — user might still be selecting
    setTimeout(() => {
      const s = window.getSelection();
      if (!s || s.isCollapsed || !s.toString().trim()) _hideIllustrateTooltip();
    }, 200);
    return;
  }

  // Only show in narrative mode
  if (_getMode() !== 'narrative') return;

  const text = sel.toString().trim();
  if (text.length < 15) return; // too short to illustrate

  // Must be within a message content area
  const anchor = sel.anchorNode;
  const contentEl = anchor?.nodeType === 3
    ? anchor.parentElement?.closest('.message-content')
    : anchor?.closest?.('.message-content');
  if (!contentEl) return;

  const msgEl = contentEl.closest('.message');
  if (!msgEl) return;

  // Get the full message content and the node ID for context
  const nodeId = msgEl.querySelector('[data-node-id]')?.dataset?.nodeId || '';
  const fullContent = contentEl.textContent || '';

  // Store selection data
  _illustrateSelection = {
    text,
    fullContent,
    nodeId,
    msgEl,
  };

  // Position the tooltip above the selection
  const range = sel.getRangeAt(0);
  const rect = range.getBoundingClientRect();
  _showIllustrateTooltip(rect);
}

function _showIllustrateTooltip(selectionRect) {
  if (!_illustrateTooltip) return;
  _illustrateTooltip.classList.remove('hidden');

  const isMobile = window.matchMedia('(max-width: 767px)').matches;

  if (isMobile) {
    // Pin to bottom-center (CSS handles left/bottom + safe-area insets). The
    // OS selection toolbar hugs the selection itself, so positioning near it
    // guarantees overlap — parking the button at the bottom sidesteps that.
    _illustrateTooltip.classList.add('mobile-pinned');
    _illustrateTooltip.style.left = '';
    _illustrateTooltip.style.top = '';
    return;
  }

  _illustrateTooltip.classList.remove('mobile-pinned');
  const ttRect = _illustrateTooltip.getBoundingClientRect();
  let left = selectionRect.left + (selectionRect.width / 2) - (ttRect.width / 2);
  let top = selectionRect.top - ttRect.height - 8;
  left = Math.max(8, Math.min(left, window.innerWidth - ttRect.width - 8));
  if (top < 8) top = selectionRect.bottom + 8;
  _illustrateTooltip.style.left = left + 'px';
  _illustrateTooltip.style.top = top + 'px';
}

function _hideIllustrateTooltip() {
  if (_illustrateTooltip) _illustrateTooltip.classList.add('hidden');
  _illustrateSelection = null;
}

async function _doIllustrateMoment(selection) {
  const { text, fullContent, msgEl } = selection;

  // Show inline progress loader on the message. The shared component
  // polls /api/image/generation-status so the user sees real stage
  // text (Composing scene prompt → Loading model → Generating step N/M
  // → Saving) instead of a generic spinner.
  const bubble = msgEl.querySelector('.message-bubble');
  const session = _getSession();
  const progress = createImageProgressLoader({
    session_id: session?.id || session?.session_id || '',
    variant: 'moment',
  });
  bubble.appendChild(progress.element);
  progress.start();

  try {
    if (!session) throw new Error('No active session');

    // Build conversation context (messages leading up to this one)
    const allMsgs = buildMessagesForAPI(session);
    const convMsgs = allMsgs
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(-4)
      .map(m => ({ role: m.role, content: m.content }));

    // Get character data from narrative module
    const activeChar = window.narrative?.activeCharacter;

    // Build the instruction: selected text as the scene focus
    const instruction = `Illustrate this specific moment from the narrative:\n\n"${text}"\n\nThis is the key scene to visualize. Show the characters in this exact moment with their current actions, expressions, and setting.`;

    // Image model/resolution/sampler/etc come from the server-side
    // image_active_settings (pushed by the image panel). Do not send stale
    // client-local mirrors — let the backend resolve from the authoritative
    // store so panel changes are always reflected.
    const res = await fetch('/api/image/generate-scene', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: session.id || session.session_id || '',
        instruction,
        messages: convMsgs,
        character_name: activeChar?.name || '',
        visual_traits: activeChar?.visualTraits || '',
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(extractErrorMessage(err, `HTTP ${res.status}`));
    }

    const data = await res.json();

    // Replace loader with the generated image, anchored to the message
    progress.stop();
    const imgContainer = document.createElement('div');
    imgContainer.className = 'illustrate-result';
    imgContainer.innerHTML = `
      <div class="illustrate-result-header">
        <span class="illustrate-result-label">Illustrated moment</span>
        <button class="illustrate-result-close" title="Remove">&times;</button>
      </div>
      <img src="${escapeHtml(data.url)}" alt="Illustrated moment" class="illustrate-result-img" loading="lazy">
      <div class="illustrate-result-excerpt">"${escapeHtml(text.slice(0, 120))}${text.length > 120 ? '...' : ''}"</div>
    `;
    imgContainer.querySelector('.illustrate-result-close').addEventListener('click', () => imgContainer.remove());
    imgContainer.querySelector('.illustrate-result-img').addEventListener('click', () => {
      // Open in lightbox if available
      if (window.openImageLightbox) {
        window.openImageLightbox(data.url);
      } else {
        window.open(data.url, '_blank');
      }
    });
    bubble.appendChild(imgContainer);

  } catch (err) {
    progress.stop();
    showToast(`Couldn't illustrate — ${err.message}`, 'error');
  }
}
