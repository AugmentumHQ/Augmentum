/* ------------------------------------------------------------------
   ChatAutocomplete — Observation Substrate consumer for chat composer.

   Inline ghost-text overlay. The suggestion appears in faded color
   immediately after the user's caret, as if it's continuing their
   text. Two acceptance gestures:

     - Tab key      → accept the full suggestion
     - Click a word → accept up to and including that word, then let
                      the user keep typing whatever comes next

   The overlay is a single absolutely-positioned <div> hosting clickable
   word spans. Caret-position measurement uses the standard mirror-div
   technique (a hidden <div> with matched styling rendering the text
   up to the caret, with a measurement marker at the end). Auto-dismiss
   conditions:

     - Substrate disabled server-side
     - Empty / failed fetch
     - User selection not at end of text (mid-edit) — keeps the overlay
       from drifting into weird positions when the user revises mid-line
     - Any non-Tab keystroke that changes the text
     - Esc / blur

   Built to be quiet: silent fail on errors, no UI noise when nothing
   matches, never blocks input.
   ------------------------------------------------------------------ */

const DEBOUNCE_MS = 200;
const MIN_PREFIX_CHARS = 6;
const FETCH_TIMEOUT_MS = 1500;

// CSS properties we have to mirror from the textarea onto the
// measurement div so layout matches exactly. Anything that affects
// text wrapping or per-character width belongs here.
const MIRROR_PROPS = [
  'boxSizing', 'width', 'height',
  'overflowX', 'overflowY',
  'borderTopWidth', 'borderRightWidth',
  'borderBottomWidth', 'borderLeftWidth',
  'borderStyle',
  'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
  'fontStyle', 'fontVariant', 'fontWeight', 'fontStretch',
  'fontSize', 'fontSizeAdjust', 'lineHeight', 'fontFamily',
  'textAlign', 'textTransform', 'textIndent',
  'textDecoration', 'letterSpacing', 'wordSpacing',
  'tabSize', 'MozTabSize',
  'whiteSpace', 'wordBreak', 'wordWrap',
];

export class ChatAutocomplete {
  /**
   * @param {HTMLTextAreaElement} textarea
   * @param {object} opts
   * @param {() => string} opts.getSurface
   * @param {() => string} opts.getMode
   */
  constructor(textarea, opts = {}) {
    this.textarea = textarea;
    this.getSurface = opts.getSurface ?? (() => 'chat');
    this.getMode = opts.getMode ?? (() => '');

    this._debounceTimer = null;
    this._abortCtrl = null;
    this._suggestion = null;   // { matchedPrefix, words: [string], anchorTextLen }
    this._overlayEl = null;
    this._mirrorEl = null;

    this._wire();
  }

  // ── Wiring ─────────────────────────────────────────────────────────

  _wire() {
    this.textarea.addEventListener('input', () => this._scheduleFetch());
    this.textarea.addEventListener('keydown', (e) => this._onKeyDown(e));
    this.textarea.addEventListener('blur', () => this._dismiss());
    this.textarea.addEventListener('scroll', () => this._reposition());
    // Re-position on viewport resize — the mirror element needs to be
    // re-measured if anyone's restyled the textarea.
    window.addEventListener('resize', this._onResizeBound = () => this._reposition());
  }

  _onKeyDown(e) {
    if (e.key === 'Tab' && this._suggestion) {
      e.preventDefault();
      this._acceptAll();
      return;
    }
    // Esc dismisses without inserting.
    if (e.key === 'Escape') {
      this._dismiss();
      return;
    }
    // Any other key that produces input will trigger the 'input' event
    // and our debounced fetch will re-evaluate. We don't dismiss
    // proactively here — the fetch will replace or dismiss based on
    // the new text.
  }

  // ── Fetch path ─────────────────────────────────────────────────────

  _scheduleFetch() {
    clearTimeout(this._debounceTimer);
    const text = this.textarea.value ?? '';
    // Only show the overlay when the selection is at the end of the
    // text. Mid-edit positioning is fiddly and not the common case
    // for chat composers (most editing happens by appending).
    if (
      text.length < MIN_PREFIX_CHARS
      || this.textarea.selectionStart !== text.length
      || this.textarea.selectionEnd !== text.length
    ) {
      this._dismiss();
      return;
    }
    this._debounceTimer = setTimeout(() => this._fetch(text), DEBOUNCE_MS);
  }

  async _fetch(text) {
    if (this._abortCtrl) {
      try { this._abortCtrl.abort(); } catch { /* ignore */ }
    }
    this._abortCtrl = new AbortController();
    const ctrl = this._abortCtrl;
    const timeoutId = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);

    try {
      const url = new URL('/api/observation/complete', window.location.origin);
      url.searchParams.set('prefix', text);
      url.searchParams.set('surface', this.getSurface() || 'chat');
      const mode = this.getMode();
      if (mode) url.searchParams.set('mode', mode);

      const resp = await fetch(url.toString(), { signal: ctrl.signal });
      clearTimeout(timeoutId);
      if (!resp.ok) { this._dismiss(); return; }
      const data = await resp.json();

      if (!data || !data.substrate_enabled) { this._dismiss(); return; }
      const top = (data.suggestions || [])[0];
      if (!top || !top.continuation) { this._dismiss(); return; }

      // Race guard — the user may have kept typing while we were
      // waiting. Don't show a suggestion for stale input.
      if (this.textarea.value !== text) return;

      this._show({
        matchedPrefix: data.matched_prefix || '',
        continuation: top.continuation,
        anchorTextLen: text.length,
      });
    } catch (err) {
      if (err?.name !== 'AbortError') {
        console.debug('[autocomplete] fetch failed', err);
      }
      this._dismiss();
    }
  }

  // ── Display ────────────────────────────────────────────────────────

  _show({ matchedPrefix, continuation, anchorTextLen }) {
    // Split continuation into words while preserving the whitespace
    // separators — clicking on word N must insert words 1..N plus the
    // separators between them, no more.
    const tokens = this._tokenize(continuation);
    if (tokens.length === 0) { this._dismiss(); return; }

    this._suggestion = { matchedPrefix, tokens, anchorTextLen };

    if (!this._overlayEl) this._buildOverlay();
    this._renderOverlay();
    this._reposition();
  }

  _tokenize(text) {
    // Split into alternating word and whitespace tokens. Each "word"
    // token gets its own clickable span; whitespace renders inline
    // between them without being clickable.
    // Regex matches sequences of non-whitespace or sequences of
    // whitespace. Order-preserving.
    const out = [];
    const re = /\s+|\S+/g;
    let m;
    while ((m = re.exec(text)) !== null) {
      out.push({
        text: m[0],
        isWord: !/^\s+$/.test(m[0]),
      });
    }
    return out;
  }

  _buildOverlay() {
    this._overlayEl = document.createElement('div');
    this._overlayEl.className = 'chat-autocomplete-overlay';
    this._overlayEl.setAttribute('role', 'status');
    this._overlayEl.setAttribute('aria-live', 'polite');
    // Land in the wrapper so the absolute positioning is relative to
    // the same containing block as the textarea.
    const wrapper = this.textarea.parentElement;
    if (wrapper) {
      // Ensure the wrapper is a positioning context for our absolutely-
      // positioned overlay. If something else already set position,
      // leave it alone.
      const cs = window.getComputedStyle(wrapper);
      if (cs.position === 'static') {
        wrapper.style.position = 'relative';
      }
      wrapper.appendChild(this._overlayEl);
    }
  }

  _renderOverlay() {
    if (!this._overlayEl || !this._suggestion) return;
    this._overlayEl.replaceChildren();
    this._overlayEl.style.display = '';

    this._suggestion.tokens.forEach((tok, idx) => {
      const span = document.createElement('span');
      span.textContent = tok.text;
      if (tok.isWord) {
        span.className = 'chat-autocomplete-word';
        // Click to accept up to and including this word + any
        // trailing whitespace token (so the user's caret lands AFTER
        // the space, ready for them to keep typing).
        span.addEventListener('mousedown', (e) => {
          // mousedown rather than click so we beat the textarea's
          // blur from the click stealing focus.
          e.preventDefault();
          this._acceptThrough(idx);
        });
      } else {
        span.className = 'chat-autocomplete-space';
      }
      this._overlayEl.appendChild(span);
    });
  }

  // ── Acceptance paths ───────────────────────────────────────────────

  _acceptAll() {
    if (!this._suggestion) return;
    this._acceptThrough(this._suggestion.tokens.length - 1);
  }

  _acceptThrough(wordIdx) {
    if (!this._suggestion) return;
    // Slice tokens up to and including the chosen word AND any
    // trailing whitespace token immediately after it (so the caret
    // sits past the boundary). The user might have clicked on a
    // whitespace token directly — that's effectively "accept the
    // previous word's boundary," handled by including the clicked
    // token itself.
    let end = wordIdx;
    const nextTok = this._suggestion.tokens[wordIdx + 1];
    if (nextTok && !nextTok.isWord) end = wordIdx + 1;

    const slice = this._suggestion.tokens.slice(0, end + 1);
    let insertion = slice.map(t => t.text).join('');

    // If the user's text doesn't end with a space and the insertion
    // doesn't start with one, glue a separator in — matches the chip
    // version's behavior and keeps "hello"+"world" from joining.
    const before = this.textarea.value;
    if (
      before.length > 0
      && !before.endsWith(' ')
      && !insertion.startsWith(' ')
    ) {
      insertion = ' ' + insertion;
    }

    this.textarea.value = before + insertion;
    const end_ = this.textarea.value.length;
    this.textarea.setSelectionRange(end_, end_);
    this.textarea.focus();
    // Trigger input so autosize + prewarm + autocomplete re-fetch fire.
    this.textarea.dispatchEvent(new Event('input', { bubbles: true }));
    this._dismiss();
  }

  // ── Positioning (caret-pixel measurement via mirror div) ───────────

  _reposition() {
    if (!this._overlayEl || !this._suggestion) return;
    const coords = this._caretCoords(this._suggestion.anchorTextLen);
    if (!coords) return;
    // The textarea's content area is offset from the wrapper by the
    // textarea's own offsetLeft/offsetTop. The caret coords are
    // relative to the textarea's content box, so we add textarea's
    // own offset within the wrapper to land the overlay correctly.
    const ta = this.textarea;
    this._overlayEl.style.top = (ta.offsetTop + coords.top - ta.scrollTop) + 'px';
    this._overlayEl.style.left = (ta.offsetLeft + coords.left - ta.scrollLeft) + 'px';
    this._overlayEl.style.height = coords.height + 'px';
  }

  _caretCoords(position) {
    // Standard "create a hidden mirror div, replicate textarea styling,
    // copy text up to caret, place a marker span at the end, read its
    // offset" pattern. Cheap enough to do on every keystroke for the
    // single-position case we need.
    const ta = this.textarea;
    if (!this._mirrorEl) {
      this._mirrorEl = document.createElement('div');
      // Off-screen but in flow — needs to participate in layout for
      // the marker offsets to be meaningful.
      this._mirrorEl.style.position = 'absolute';
      this._mirrorEl.style.visibility = 'hidden';
      this._mirrorEl.style.top = '-9999px';
      this._mirrorEl.style.left = '0';
      document.body.appendChild(this._mirrorEl);
    }
    const cs = window.getComputedStyle(ta);
    for (const prop of MIRROR_PROPS) {
      this._mirrorEl.style[prop] = cs[prop];
    }
    // Textareas treat newlines as line breaks visually; the mirror
    // <div> needs white-space: pre-wrap so it does the same.
    this._mirrorEl.style.whiteSpace = 'pre-wrap';
    this._mirrorEl.style.wordWrap = 'break-word';

    const value = ta.value.substring(0, position);
    this._mirrorEl.textContent = value;
    const marker = document.createElement('span');
    // Inserting an invisible character ensures the span has a
    // measurable bounding box even when the caret sits at a line
    // break or empty position.
    marker.textContent = '​';
    this._mirrorEl.appendChild(marker);

    const top = marker.offsetTop;
    const left = marker.offsetLeft;
    const height = parseFloat(cs.lineHeight) || (parseFloat(cs.fontSize) * 1.4);
    // Clean up immediately so DOM doesn't grow.
    this._mirrorEl.removeChild(marker);
    return { top, left, height };
  }

  // ── Teardown ───────────────────────────────────────────────────────

  _dismiss() {
    this._suggestion = null;
    if (this._overlayEl) {
      this._overlayEl.replaceChildren();
      this._overlayEl.style.display = 'none';
    }
    if (this._abortCtrl) {
      try { this._abortCtrl.abort(); } catch { /* ignore */ }
      this._abortCtrl = null;
    }
    clearTimeout(this._debounceTimer);
  }

  destroy() {
    this._dismiss();
    if (this._overlayEl?.parentElement) {
      this._overlayEl.parentElement.removeChild(this._overlayEl);
    }
    if (this._mirrorEl?.parentElement) {
      this._mirrorEl.parentElement.removeChild(this._mirrorEl);
    }
    this._overlayEl = null;
    this._mirrorEl = null;
    if (this._onResizeBound) {
      window.removeEventListener('resize', this._onResizeBound);
    }
  }
}
