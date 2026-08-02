/**
 * stream-render.js — one incremental streaming renderer for every chat mode.
 *
 * Both the passthrough/narrative renderer (chat/renderer.js) and the coder
 * conversation view (coder-conversation.js) stream model output into the DOM.
 * They used to do it two different ways and BOTH degraded to O(n²):
 *
 *   - renderer.js had a stable/active split + rAF coalescing, but the
 *     "active" suffix could not advance past an OPEN ``` fence
 *     (``findStableBoundary`` refuses to promote across an unbalanced
 *     fence). So the moment a code block opened, the whole growing block
 *     stayed in the active region and was re-parsed via ``renderMarkdown``
 *     AND re-highlighted from scratch every animation frame — because
 *     ``activeEl.innerHTML = …`` destroys the DOM each frame, wiping
 *     hljs's ``data-highlighted`` self-skip flags. A long file-write at a
 *     fast tok/s pinned the main thread for tens of seconds.
 *
 *   - coder-conversation.js had none of that: every delta did
 *     ``innerHTML = renderMarkdown(WHOLE raw)`` synchronously at the
 *     network cadence. Pure O(n²), no coalescing — the worst-reported case.
 *
 * This module is the single source of the split-render algorithm. The new
 * piece is incremental OPEN-FENCE handling: while a code fence is open the
 * code body is appended as plain text nodes (O(delta), no per-frame
 * highlight). When the fence closes it promotes to the stable region and is
 * highlighted once. Highlighting on the stream path is deferred to the idle
 * queue so it never blocks input.
 *
 * Callers own a per-stream state object (``newSplitState()``), reset it when
 * a new streaming message starts, and call ``renderStreamSplit`` from a
 * rAF-coalesced flush.
 */

import { renderMarkdown as fullRenderMarkdown, highlightCodeDeferred } from './markdown.js';
import { findStableBoundary, findOpenFence, findForcedBoundary } from './stream-fence.js';

export { findStableBoundary, findOpenFence, findForcedBoundary }; // re-exported for callers/tests

const _SCAFFOLD = '<div class="stream-stable"></div><div class="stream-active"></div>';

/** Force-promote a non-fence active suffix once it grows past this many chars
 *  with no ``\n\n`` to settle it — bounds the per-frame re-parse so a giant
 *  single paragraph at fast tok/s can't degrade to O(n²). See findForcedBoundary. */
const _ACTIVE_SOFT_CAP = 6000;

/** Fresh per-stream bookkeeping. One per active streaming message. */
export function newSplitState() {
  return {
    stableEnd: 0,        // raw index (exclusive) already committed to stable
    stableCodeCount: 0,  // <pre><code> count last highlighted in stable
    fence: null,         // active open-fence sub-render, or null
  };
}

/** Ensure the ``.response-body`` + stable/active scaffold exists under
 *  ``contentEl``. Resets ``state`` when it has to (re)build. Returns
 *  ``{ stableEl, activeEl }``. */
function _ensureScaffold(contentEl, state) {
  let responseBody = contentEl.querySelector('.response-body');
  if (!responseBody) {
    responseBody = document.createElement('div');
    responseBody.className = 'response-body streaming-cursor';
    responseBody.innerHTML = _SCAFFOLD;
    contentEl.appendChild(responseBody);
    state.stableEnd = 0;
    state.stableCodeCount = 0;
    state.fence = null;
  }
  let stableEl = responseBody.querySelector(':scope > .stream-stable');
  let activeEl = responseBody.querySelector(':scope > .stream-active');
  if (!stableEl || !activeEl) {
    // Stale single-block layout (older path / replaceStreamedContent).
    responseBody.innerHTML = _SCAFFOLD;
    stableEl = responseBody.querySelector(':scope > .stream-stable');
    activeEl = responseBody.querySelector(':scope > .stream-active');
    state.stableEnd = 0;
    state.stableCodeCount = 0;
    state.fence = null;
  }
  return { stableEl, activeEl };
}

/**
 * Render an active region that contains an OPEN code fence incrementally.
 * Prose before the fence re-renders only when it changes (it's frozen once
 * the fence opens). The code body grows by appending text nodes — O(delta),
 * no markdown re-parse, no per-frame highlight. Highlight happens once when
 * the block closes and promotes to stable.
 */
function _renderOpenFence(activeEl, activeRaw, open, state, mdOpts, renderMarkdown) {
  const absOpen = state.stableEnd + open.openIdx; // stable identity for this fence
  let f = state.fence;
  if (!f || f.absOpen !== absOpen) {
    activeEl.innerHTML = '';
    const proseEl = document.createElement('div');
    proseEl.className = 'stream-active-prose';
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    if (open.lang) code.className = `language-${open.lang}`;
    pre.appendChild(code);
    activeEl.appendChild(proseEl);
    activeEl.appendChild(pre);
    f = state.fence = { absOpen, proseEl, codeEl: code, proseLen: -1, renderedLen: 0 };
  }
  // Prologue (prose before the opener). Short and frozen once fence opens.
  const prologue = activeRaw.slice(0, open.openIdx);
  if (prologue.length !== f.proseLen) {
    f.proseEl.innerHTML = prologue ? renderMarkdown(prologue, mdOpts) : '';
    f.proseLen = prologue.length;
  }
  // Code body — append only the new tail. textContent escapes for us.
  const body = activeRaw.slice(open.bodyStart);
  if (body.length > f.renderedLen) {
    f.codeEl.appendChild(document.createTextNode(body.slice(f.renderedLen)));
    f.renderedLen = body.length;
  }
}

/**
 * Incrementally render ``raw`` into ``contentEl`` using a stable/active
 * split. Settled paragraphs are parsed once and appended to the stable
 * region; the still-streaming suffix re-renders cheaply; an open code fence
 * is handled append-only. Mutates ``state`` and the DOM.
 *
 * @param {HTMLElement} contentEl - container that owns ``.response-body``
 * @param {string} raw - full accumulated raw text so far
 * @param {object} state - per-stream bookkeeping from ``newSplitState()``
 * @param {object} [opts] - { mode, narrativePanelsCollapsed, highlightHooks }
 */
export function renderStreamSplit(contentEl, raw, state, opts = {}) {
  // Markdown flavor is injectable so every AI-help surface (chat, coder,
  // browse, studio, …) shares this one incremental ENGINE while keeping its
  // own renderer (full chat markdown vs a compact one). Defaults to the full
  // chat renderer.
  const renderMarkdown = opts.renderMarkdown || fullRenderMarkdown;
  const mdOpts = {
    mode: opts.mode,
    narrativePanelsCollapsed: opts.narrativePanelsCollapsed,
    compact: opts.compact === true,  // compact surfaces drop the chat-only code toolbar
  };
  const { stableEl, activeEl } = _ensureScaffold(contentEl, state);

  // 1) Promote balanced content to the stable region (append-only so
  //    already-highlighted code keeps its hljs flag and isn't re-walked).
  const newBoundary = findStableBoundary(raw, state.stableEnd);
  if (newBoundary > state.stableEnd) {
    _promoteToStable(stableEl, raw.slice(state.stableEnd, newBoundary), state, mdOpts, renderMarkdown, opts);
    state.stableEnd = newBoundary;
    state.fence = null; // anything promoted invalidates an open-fence sub-render
  }

  // 2) Active region.
  let activeRaw = raw.slice(state.stableEnd);
  const open = findOpenFence(activeRaw);
  if (open) {
    _renderOpenFence(activeEl, activeRaw, open, state, mdOpts, renderMarkdown);
  } else {
    // No open fence. Normally the suffix is short and cheap to re-render
    // wholesale — but a long paragraph with no `\n\n` (so findStableBoundary
    // never advances) would re-parse the whole growing string every frame:
    // O(n²). The active-tail cap force-promotes a SAFE prose prefix to bound
    // it. findForcedBoundary only fires on flat prose, so it can never split a
    // list/table/etc. — the worst case is a soft line-break turned paragraph.
    state.fence = null;
    const forced = findForcedBoundary(activeRaw, opts.activeSoftCap || _ACTIVE_SOFT_CAP);
    if (forced > 0) {
      _promoteToStable(stableEl, activeRaw.slice(0, forced), state, mdOpts, renderMarkdown, opts);
      state.stableEnd += forced;
      activeRaw = activeRaw.slice(forced);
    }
    activeEl.innerHTML = activeRaw ? renderMarkdown(activeRaw, mdOpts) : '';
    if (activeRaw && activeEl.querySelector('pre code')) {
      highlightCodeDeferred(activeEl, opts.highlightHooks);
    }
  }
}

/** Parse ``chunk`` once and APPEND it to the stable region (never re-touch
 *  settled DOM), deferring code highlight to idle. Shared by the natural
 *  paragraph-boundary promotion and the active-tail cap. */
function _promoteToStable(stableEl, chunk, state, mdOpts, renderMarkdown, opts) {
  const tmp = document.createElement('div');
  tmp.innerHTML = renderMarkdown(chunk, mdOpts);
  const frag = document.createDocumentFragment();
  while (tmp.firstChild) frag.appendChild(tmp.firstChild);
  stableEl.appendChild(frag);
  const codeCount = stableEl.querySelectorAll('pre code').length;
  if (codeCount > state.stableCodeCount) {
    // Deferred: colors fill in on the idle queue, never blocking input.
    highlightCodeDeferred(stableEl, opts.highlightHooks);
    state.stableCodeCount = codeCount;
  }
}

/**
 * Per-block streaming renderer for the "SSE ``data:`` delta into one block"
 * pattern used by the browse/studio/youtube AI surfaces (and anywhere else a
 * single element receives a growing text stream). Encapsulates the per-stream
 * split state AND rAF coalescing so call sites shrink to ``r.render(fullText)``
 * per delta — no more ``el.innerHTML = render(WHOLE text)`` every chunk.
 *
 * @param {HTMLElement} contentEl - the block to render into
 * @param {object} [opts] - forwarded to ``renderStreamSplit``; plus optional
 *   ``onFlush()`` run after each coalesced render (e.g. keep-in-view scroll).
 * @returns {{ render(fullText: string): void, reset(): void }}
 */
export function makeStreamRenderer(contentEl, opts = {}) {
  const state = newSplitState();
  let scheduled = false;
  let latest = '';
  const flush = () => {
    scheduled = false;
    renderStreamSplit(contentEl, latest, state, opts);
    if (opts.onFlush) opts.onFlush();
  };
  return {
    render(fullText) {
      latest = fullText || '';
      if (scheduled) return;
      scheduled = true;
      if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
        window.requestAnimationFrame(flush);
      } else {
        flush(); // test / non-browser host
      }
    },
    reset() {
      state.stableEnd = 0;
      state.stableCodeCount = 0;
      state.fence = null;
    },
  };
}
