/*
 * narration-bar.js — READER-DRIVEN comic narration. The user is the clock.
 *
 * The sibling module `narration-player.js` is the other model: a self-contained
 * viewer that owns the page image, owns a page cursor, and advances when a
 * page's audio ends. That is correct for CAST — a TV has no reader and nobody
 * scrolling it, so audio has to drive the camera. It is wrong for the in-app
 * reader, for three reasons found in use (2026-07-25):
 *
 *   1. It mounted OVER the reader as `comic-narration-overlay`, so the user
 *      lost their own navigation — they couldn't turn pages or scroll.
 *   2. `onEnded -> playIndex(idx + 1)` turned pages by itself, on the audio's
 *      schedule rather than the reader's.
 *   3. Its page cursor was a SECOND source of truth. Read to page 40, close
 *      the overlay, and you were back on the reader's page 1 — desynced by
 *      construction, not by bug.
 *
 * This module inverts all three: it renders no page and owns no cursor. The
 * reader tells it which page is visible via `setPage()`, and it plays that
 * page's audio. When the audio ends it STOPS and waits for the user to turn.
 * There is exactly one page cursor in the reader, so (3) cannot happen.
 *
 * A side effect worth naming: because nothing here draws on the page, per-line
 * `bbox` is no longer needed for anything but the caption's own highlight —
 * which it doesn't have. The VLM OCR engine returns `bbox: null` for every
 * line (VLMs confabulate coordinates), and under this model that costs nothing.
 *
 * payload: same shape the narration API returns (see narration-player.js).
 *   Page numbers in `pages[].page` are 0-INDEXED; the reader's are 1-indexed.
 *   The seam is confined to `_forPage()`.
 * opts: {
 *   pollUrl?: string,          // poll for newly-synthesized pages
 *   pollMs?: number,
 *   requestPage?: (page1) => void,  // auto-advance asks the READER to turn
 *   isContinuous?: () => boolean,   // true in webtoon: queue turns, don't cut
 *   onClose?: () => void,
 * }
 */

const _NS = 'comic-narration-bar';

// How long the reader's page has to hold still before we treat it as "where
// the user is". Long enough to ride out scroll jitter and a fast flick,
// short enough that a deliberate page turn still feels immediate.
const SETTLE_MS = 400;

export function mountNarrationBar(container, payload, opts = {}) {
  if (!container || !payload) return null;
  let pages = Array.isArray(payload.pages) ? payload.pages.slice() : [];

  const pollUrl = opts.pollUrl || '';
  const pollMs = opts.pollMs || 2500;
  let status = payload.status || 'running';

  const el = document.createElement('div');
  el.className = _NS;

  const playBtn = document.createElement('button');
  playBtn.type = 'button';
  playBtn.className = `${_NS}-play`;
  playBtn.textContent = '▶';
  playBtn.setAttribute('aria-label', 'Play narration');

  const caption = document.createElement('div');
  caption.className = `${_NS}-caption`;

  const statusEl = document.createElement('div');
  statusEl.className = `${_NS}-status`;

  // Auto-advance is OPT-IN and off by default. When on, the end of a page's
  // audio asks the READER to turn the page — it never turns one itself, so
  // the reader stays the only thing that moves the cursor.
  const autoWrap = document.createElement('label');
  autoWrap.className = `${_NS}-auto`;
  const autoBox = document.createElement('input');
  autoBox.type = 'checkbox';
  const autoText = document.createElement('span');
  autoText.textContent = 'Auto-turn';
  autoWrap.appendChild(autoBox);
  autoWrap.appendChild(autoText);
  autoWrap.title = 'When a page finishes, turn to the next one automatically';

  const close = document.createElement('button');
  close.type = 'button';
  close.className = `${_NS}-close`;
  close.textContent = '✕';
  close.setAttribute('aria-label', 'Stop narration');

  const audio = document.createElement('audio');
  audio.className = `${_NS}-audio`;
  audio.preload = 'auto';

  el.appendChild(playBtn);
  const mid = document.createElement('div');
  mid.className = `${_NS}-mid`;
  mid.appendChild(caption);
  mid.appendChild(statusEl);
  el.appendChild(mid);
  el.appendChild(autoWrap);
  el.appendChild(close);
  el.appendChild(audio);
  container.appendChild(el);

  let destroyed = false;
  let pollTimer = null;
  let preloader = null;
  // The page the reader last told us about (1-indexed), and the page whose
  // audio is currently loaded. They differ while we wait for synthesis.
  let wantPage = 0;
  let loadedPage = 0;
  let paused = false;      // user pressed pause; don't auto-start on turn
  let ended = false;       // current page's audio ran out
  let settleTimer = null;
  // A page the reader has moved to while the CURRENT page is still speaking.
  // In continuous (webtoon) reading the page boundary is just a position in a
  // scroll — crossing it is not a statement that you're done listening, and
  // cutting the audio off mid-sentence because a few pixels of the next page
  // came into view is the single most jarring thing this bar can do. So the
  // turn is remembered and applied when the current page finishes.
  // Zero means "nothing waiting".
  let queuedPage = 0;
  // Pages whose audio has already been started once. Auto-play is for FORWARD
  // progress only: a page you've already heard loads silently and waits for
  // play. Without this, nudging the scroll back over the previous page's
  // boundary — trivially easy in webtoon mode, where the observer fires on a
  // few pixels — restarts that page's narration on its own, which reads as the
  // reader talking over itself. Re-listening is a deliberate act (press play),
  // not something scrolling should trigger.
  const played = new Set();

  function _forPage(page1) {
    return pages.find((p) => (p.page || 0) === page1 - 1) || null;
  }

  function _lines() {
    const p = _forPage(loadedPage);
    return (p && Array.isArray(p.lines)) ? p.lines : [];
  }

  function setStatus(text) { statusEl.textContent = text || ''; }

  /** Is the reader in a continuous-scroll mode right now?
   *
   * Asked live rather than captured at mount, because the mode is a key press
   * away (W) and the bar outlives the switch. Defaults to false: paged reading
   * is the safe assumption, since there a page turn IS a deliberate act and
   * interrupting is the correct response.
   */
  function isContinuous() {
    try { return !!opts.isContinuous?.(); } catch { return false; }
  }

  /** Audio that is actually making sound right now. */
  function isSpeaking() {
    return !!audio.src && !audio.paused && !audio.ended && !paused;
  }

  function setPlayIcon() {
    const playing = !audio.paused && !audio.ended && !!audio.src;
    playBtn.textContent = playing ? '⏸' : '▶';
    playBtn.setAttribute('aria-label', playing ? 'Pause narration' : 'Play narration');
  }

  function preloadNext() {
    const n = _forPage(wantPage + 1);
    if (!n || !n.audio_url) return;
    preloader = new Audio();
    preloader.preload = 'auto';
    preloader.src = n.audio_url;
  }

  // Load + play the page the reader is showing. If it isn't synthesized yet,
  // hold with an honest message; polling retries when it lands.
  function playWanted({ userInitiated = false } = {}) {
    if (destroyed || !wantPage) return;
    const p = _forPage(wantPage);
    if (!p || !p.audio_url) {
      loadedPage = 0;
      audio.removeAttribute('src');
      caption.textContent = '';
      // A page that EXISTS with no audio is finished, not pending: the synth
      // job records splash art and unreadable pages as real entries with an
      // empty artifact_id. Only a MISSING entry is still queued. Collapsing
      // those two into one message left silent pages saying "Synthesizing…"
      // forever while the pages after them played — the wait never resolves,
      // because nothing further is coming for that page.
      if (status === 'failed' && !p) setStatus('Narration failed.');
      else if (p) setStatus(`No narration for page ${wantPage} — no text on this page.`);
      else if (status === 'done') setStatus(`No narration for page ${wantPage}.`);
      else setStatus(`Synthesizing page ${wantPage}…`);
      ensurePolling();
      setPlayIcon();
      return;
    }
    // Two reasons to load a page but NOT speak it: the user paused (turning a
    // page while paused is browsing, not resuming), or they've come back to a
    // page they already heard (scrolling back is re-reading the art, not a
    // request to re-hear the dialogue). Either way the page is armed and one
    // press plays it.
    const replay = played.has(wantPage);
    if ((paused || replay) && !userInitiated) {
      loadedPage = wantPage;
      audio.src = p.audio_url;
      ended = false;
      caption.textContent = '';
      setStatus(replay && !paused
        ? `Page ${wantPage} — already read. Press play to hear it again.`
        : `Page ${wantPage} ready — press play.`);
      setPlayIcon();
      return;
    }
    loadedPage = wantPage;
    played.add(wantPage);
    ended = false;
    audio.src = p.audio_url;
    try { audio.currentTime = 0; } catch { /* not seekable yet */ }
    caption.textContent = '';
    setStatus(`Page ${wantPage}`);
    audio.play().catch(() => {
      // Autoplay gate — the page is loaded, the user can press play.
      setStatus(`Page ${wantPage} ready — press play.`);
      setPlayIcon();
    });
    preloadNext();
  }

  /** The reader calls this whenever the visible page changes.
   *
   * Debounced, because "the visible page" is a continuous quantity during a
   * scroll: the webtoon observer reports every page the viewport crosses, and
   * a fast flick through five pages would otherwise start and abort five
   * narrations. We act on where the user LANDS, not on everything they passed.
   */
  function setPage(page1, { immediate = false } = {}) {
    const n = Number(page1) || 0;
    if (!n || destroyed) return;
    if (n === wantPage) {
      // Scrolled away and back again before settling — the pending turn is no
      // longer where the user is, so drop it rather than let it fire. Same for
      // a queued turn: scrolling back to the page that's speaking means "stay".
      if (settleTimer) { clearTimeout(settleTimer); settleTimer = null; }
      if (queuedPage) { queuedPage = 0; setStatus(`Page ${wantPage}`); }
      return;
    }
    if (settleTimer) clearTimeout(settleTimer);

    const apply = () => {
      settleTimer = null;
      if (destroyed || n === wantPage) return;
      // Continuous mode: let the current page finish its sentence. The turn is
      // parked in queuedPage and onEnded picks it up. Only ever defers to a
      // page we aren't already on, and only while audio is genuinely playing —
      // a paused or finished page has nothing to protect, so it switches now.
      if (!immediate && isContinuous() && isSpeaking()) {
        queuedPage = n;
        setStatus(`Page ${wantPage} — page ${n} next`);
        return;
      }
      queuedPage = 0;
      wantPage = n;
      audio.pause();
      playWanted();
      setPlayIcon();
    };

    if (immediate) { apply(); return; }
    settleTimer = setTimeout(apply, SETTLE_MS);
  }

  function onTime() {
    const lines = _lines();
    if (!lines.length) return;
    const ms = audio.currentTime * 1000;
    for (const b of lines) {
      if (ms >= (b.audio_start_ms || 0) && ms < (b.audio_end_ms || 0)) {
        if (caption.textContent !== (b.text || '')) {
          caption.textContent = b.text || '';
          caption.dataset.kind = b.kind || 'speech';
        }
        return;
      }
    }
  }

  function onEnded() {
    ended = true;
    setPlayIcon();
    // A turn the user already made, held back so this page could finish. It
    // outranks auto-advance: the reader is ALREADY on that page, so asking it
    // to turn would fight the user's own scrolling. Just speak what they're
    // looking at.
    if (queuedPage) {
      const next = queuedPage;
      queuedPage = 0;
      wantPage = next;
      playWanted();
      setPlayIcon();
      return;
    }
    if (autoBox.checked && typeof opts.requestPage === 'function') {
      const next = wantPage + 1;
      setStatus('Turning the page…');
      opts.requestPage(next);
      return;
    }
    setStatus('Turn the page to continue.');
  }

  // ---- polling for newly-synthesized pages -------------------------------
  function mergePages(incoming) {
    if (!Array.isArray(incoming)) return false;
    let added = false;
    const have = new Set(pages.map((p) => p.page));
    for (const p of incoming) {
      if (!have.has(p.page)) { pages.push(p); have.add(p.page); added = true; }
    }
    if (added) pages.sort((a, b) => a.page - b.page);
    return added;
  }

  async function poll() {
    pollTimer = null;
    if (destroyed || !pollUrl) return;
    try {
      const r = await fetch(pollUrl, { credentials: 'same-origin' });
      if (r.ok) {
        const data = await r.json();
        status = data.status || status;
        const added = mergePages(data.pages || []);
        // Only resume if we're still holding for the page the reader wants —
        // the user may have turned away while we waited.
        if ((added || status === 'done' || status === 'failed')
            && wantPage && loadedPage !== wantPage) {
          playWanted();
        }
      }
    } catch { /* transient — keep polling */ }
    if (!destroyed && status !== 'done' && status !== 'failed') {
      pollTimer = setTimeout(poll, pollMs);
    }
  }

  function ensurePolling() {
    if (!pollTimer && pollUrl && status !== 'done' && status !== 'failed') {
      pollTimer = setTimeout(poll, pollMs);
    }
  }

  // ---- wiring ------------------------------------------------------------
  audio.addEventListener('timeupdate', onTime);
  audio.addEventListener('ended', onEnded);
  audio.addEventListener('play', setPlayIcon);
  audio.addEventListener('pause', setPlayIcon);
  playBtn.addEventListener('click', () => {
    if (!audio.src || (ended && loadedPage === wantPage)) {
      paused = false;
      playWanted({ userInitiated: true });   // replay this page / start it
      return;
    }
    if (audio.paused) { paused = false; audio.play().catch(() => {}); }
    else {
      paused = true;
      audio.pause();
      // Pausing ends the sentence the queue was protecting, and `ended` will
      // now never fire — so a turn parked behind it would be stranded until
      // the user scrolled again. Apply it now: they're looking at that page.
      if (queuedPage) {
        wantPage = queuedPage;
        queuedPage = 0;
        playWanted();   // honours `paused` — loads the page, waits for play
      }
    }
    setPlayIcon();
  });
  close.addEventListener('click', () => { opts.onClose?.(); });

  if (status !== 'done' && status !== 'failed') ensurePolling();
  setPlayIcon();

  return {
    el,
    setPage,
    /** Start on whatever page the reader is currently showing. */
    start(page1) {
      wantPage = 0;
      paused = false;
      played.clear();
      queuedPage = 0;
      setPage(page1, { immediate: true });
    },
    destroy() {
      destroyed = true;
      queuedPage = 0;
      if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
      if (settleTimer) { clearTimeout(settleTimer); settleTimer = null; }
      audio.pause();
      audio.removeEventListener('timeupdate', onTime);
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('play', setPlayIcon);
      audio.removeEventListener('pause', setPlayIcon);
      audio.removeAttribute('src');
      if (preloader) { preloader.src = ''; preloader = null; }
      el.remove();
    },
  };
}

export default { mountNarrationBar };
