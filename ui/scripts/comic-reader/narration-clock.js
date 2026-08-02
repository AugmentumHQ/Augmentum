/**
 * narration-clock.js — the audio-IS-the-clock engine for voiced comics,
 * factored OUT of narration-player.js's rendering so any surface can drive
 * its own visuals from it.
 *
 * "Audio is the clock": each comic page is synthesized into its own audio
 * clip; the clock plays them in sequence, and as a page's clip advances it
 * reports which bubble is currently spoken. When a page's clip ends it
 * advances to the next page. While synthesis is still streaming, it polls
 * for newly-ready pages and holds ("waiting") if playback catches up.
 *
 * This module owns ONLY scheduling + the <audio> element. It renders
 * nothing. Surfaces subscribe via callbacks and do the drawing:
 *   - narration-player.js (in-app reader): highlights a bubble + scrolls.
 *   - cast-comic (TV): loads the page image in its own renderer + can
 *     highlight, and routes clock.audio through AudioBus at the speech
 *     tier so a music bed ducks under the narration.
 *
 * Why separate: the in-app reader and the TV reader draw pages completely
 * differently (fit-width scroller vs adaptive dual/webtoon with crop). One
 * shared renderer can't serve both, but the SCHEDULING is identical — so
 * the clock is the shared part. Mirrors music-source.js's resolve-vs-play
 * split: the clock decides WHEN + WHAT is spoken; the surface decides HOW
 * it looks.
 *
 * payload (from GET/POST /api/comic-narration/{ref}[/cast]):
 *   { pages:[{ page, audio_url, duration_ms,
 *              lines:[{order,kind,text,bbox,audio_start_ms,audio_end_ms}] }],
 *     page_url_template, total_pages, reading_direction, status }
 *
 * opts:
 *   pollUrl   — URL to poll for streaming pages (omit for a finished payload)
 *   pollMs    — poll interval (default 2500)
 *   minPageMs — guaranteed minimum on-screen time per page; a page whose audio
 *               is shorter is HELD for the remainder before advancing, so a
 *               short-narration page isn't flashed past (default 0 = off).
 *   pageCushionMs — a fixed beat inserted after every page before the next one
 *               loads, so page flips breathe rather than snap (default 0).
 *   splashMs  — dwell for a SILENT page (splash art / unreadable), which has no
 *               audio to end it (default: minPageMs, or 2500 when unset). Also
 *               the fix for such a page stalling the clock forever.
 *   onPage    — ({ page, index, lines, total, status }) new page started;
 *               `page` is 0-indexed as delivered by the server.
 *   onLine    — ({ index, line, page }) active bubble changed; index -1 clears.
 *   onWaiting — ({ status }) playback caught up to synthesis; holding.
 *   onFinish  — () end of narration reached (status done, no more pages).
 *   onStatus  — (status) synthesis status changed ('running'|'done'|'failed').
 *   onCounter — ({ humanPage, total, status }) counter data changed.
 */

/**
 * Create a narration clock. Returns a handle; call play() to start (browsers
 * gate autoplay, so the first play() should be user-initiated or best-effort).
 */
export function createNarrationClock(payload, opts = {}) {
  const noop = () => {};
  const onPage = opts.onPage || noop;
  const onLine = opts.onLine || noop;
  const onWaiting = opts.onWaiting || noop;
  const onFinish = opts.onFinish || noop;
  const onStatus = opts.onStatus || noop;
  const onCounter = opts.onCounter || noop;

  let pages = Array.isArray(payload?.pages) ? payload.pages.slice() : [];
  const totalPages = payload?.total_pages || 0;
  const pollUrl = opts.pollUrl || '';
  const pollMs = opts.pollMs || 2500;
  const minPageMs = Math.max(0, opts.minPageMs || 0);
  const pageCushionMs = Math.max(0, opts.pageCushionMs || 0);
  const splashMs = Math.max(0, opts.splashMs != null ? opts.splashMs : (minPageMs || 2500));
  let status = payload?.status || 'running';

  let idx = 0;            // index into pages[] (the playback cursor)
  let curBubble = -1;     // active line within the current page
  let waiting = false;    // playback caught up to synthesis; awaiting a page
  let destroyed = false;
  let pollTimer = null;
  let advanceTimer = null;   // page-cushion / splash-dwell hold before advancing
  let pageShownAt = 0;       // performance.now() when the current page began
  let preloader = null;

  const _now = () => (typeof performance !== 'undefined' ? performance.now() : 0);
  function clearAdvance() {
    if (advanceTimer) { clearTimeout(advanceTimer); advanceTimer = null; }
  }
  // Advance to the next page after `ms` (0 = immediately). Any hold is a single
  // scheduled hop, cancelled by destroy / a new page load.
  function scheduleAdvance(ms) {
    clearAdvance();
    if (ms <= 0) { playIndex(idx + 1); return; }
    advanceTimer = setTimeout(() => { advanceTimer = null; playIndex(idx + 1); }, ms);
  }

  // The clock owns the audio element — it's the heartbeat. Surfaces read
  // clock.audio to route it through AudioBus (speech tier) and to read
  // currentTime; they must NOT swap its src.
  const audio = document.createElement('audio');
  audio.preload = 'auto';

  function curLines() {
    return (pages[idx] && Array.isArray(pages[idx].lines)) ? pages[idx].lines : [];
  }

  function humanCounter() {
    return {
      humanPage: pages[idx] ? (pages[idx].page + 1) : '?',
      total: totalPages || (status === 'done' ? pages.length : null),
      status,
    };
  }

  function activate(bi) {
    if (bi === curBubble) return;
    curBubble = bi;
    const lines = curLines();
    if (bi < 0 || bi >= lines.length) {
      onLine({ index: -1, line: null, page: pages[idx]?.page ?? -1 });
      return;
    }
    onLine({ index: bi, line: lines[bi], page: pages[idx]?.page ?? -1 });
  }

  function bubbleAt(ms) {
    const lines = curLines();
    for (let i = 0; i < lines.length; i++) {
      const b = lines[i];
      if (ms >= (b.audio_start_ms || 0) && ms < (b.audio_end_ms || 0)) return i;
    }
    return -1;
  }

  function onTime() {
    const bi = bubbleAt(audio.currentTime * 1000);
    if (bi >= 0) activate(bi);
  }

  function preloadNext() {
    const n = pages[idx + 1];
    if (!n || !n.audio_url) return;
    preloader = new Audio();
    preloader.preload = 'auto';
    preloader.src = n.audio_url;
  }

  // Load + play pages[i]. If it isn't synthesized yet, hold and let polling
  // resume us when it lands.
  function playIndex(i) {
    if (destroyed) return;
    clearAdvance();
    if (i >= pages.length) {
      if (status === 'done') { finish(); return; }
      waiting = true;
      onWaiting({ status });
      ensurePolling();
      return;
    }
    waiting = false;
    idx = i;
    curBubble = -1;
    pageShownAt = _now();
    const p = pages[i];
    onPage({
      page: p.page || 0,
      index: i,
      lines: Array.isArray(p.lines) ? p.lines : [],
      total: totalPages || (status === 'done' ? pages.length : null),
      status,
    });
    onCounter(humanCounter());
    if (!p.audio_url) {
      // Silent page (splash art / unreadable): there's no clip to end it, so
      // hold on the art for a dwell and then advance. Without this the clock
      // would set an empty src, never receive 'ended', and stall here forever.
      try { audio.removeAttribute('src'); audio.load(); } catch { /* ignore */ }
      activate(-1);
      preloadNext();
      scheduleAdvance(splashMs);
      return;
    }
    audio.src = p.audio_url;
    audio.currentTime = 0;
    audio.play().catch(() => { /* autoplay gate — surface can offer a play button */ });
    if (curLines().length) activate(0);
    preloadNext();
  }

  function onEnded() {
    // Guarantee the page was on screen for at least minPageMs, then add the
    // fixed cushion — so a one-line page gets a real beat instead of snapping
    // to the next flip the instant its short clip ends.
    const shown = Math.max(0, _now() - pageShownAt);
    const hold = Math.max(0, minPageMs - shown) + pageCushionMs;
    scheduleAdvance(hold);
  }

  function finish() {
    onFinish();
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
    if (destroyed || !pollUrl) return;
    try {
      const r = await fetch(pollUrl, { credentials: 'same-origin' });
      if (r.ok) {
        const data = await r.json();
        const prevStatus = status;
        status = data.status || status;
        if (status !== prevStatus) onStatus(status);
        const added = mergePages(data.pages || []);
        onCounter(humanCounter());
        // While waiting, the page we want is always the one after the last
        // we played (idx + 1). Resume when it lands, or when synthesis ends.
        if (waiting && (added || status === 'done' || status === 'failed')) {
          if (pages.length > idx + 1 || status === 'done' || status === 'failed') {
            playIndex(idx + 1);
          }
        }
      }
    } catch { /* transient — keep polling */ }
    if (!destroyed && status !== 'done' && status !== 'failed') {
      pollTimer = setTimeout(poll, pollMs);
    } else {
      pollTimer = null;
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

  // If synthesis isn't done, keep polling from the start so the buffer grows
  // ahead of playback.
  if (status !== 'done' && status !== 'failed') ensurePolling();

  return {
    audio,
    /** Start (or restart) playback from the first page. */
    play: () => playIndex(0),
    /** Resume the current clip without restarting. */
    resume: () => audio.play().catch(() => {}),
    pause: () => audio.pause(),
    togglePause: () => { audio.paused ? audio.play().catch(() => {}) : audio.pause(); },
    isPaused: () => audio.paused,
    /** Jump the cursor to a specific pages[] index (bounded). */
    playPageIndex: (i) => playIndex(Math.max(0, Math.min(i, pages.length))),
    currentPageInfo: () => ({
      page: pages[idx]?.page ?? -1,
      index: idx,
      lines: curLines(),
      total: totalPages || (status === 'done' ? pages.length : null),
      status,
    }),
    counter: humanCounter,
    destroy: () => {
      destroyed = true;
      if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
      clearAdvance();
      audio.pause();
      audio.removeEventListener('timeupdate', onTime);
      audio.removeEventListener('ended', onEnded);
      if (preloader) { preloader.src = ''; preloader = null; }
      try { audio.removeAttribute('src'); audio.load(); } catch { /* ignore */ }
    },
  };
}

export default { createNarrationClock };
