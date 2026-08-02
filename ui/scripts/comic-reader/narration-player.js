/*
 * narration-player.js — read-along scroll player for voiced comics, STREAMING
 * edition. Audio is the clock: as a page's narration plays, the player keeps
 * the FULL page in view (fit-width, art never hidden) and gently scrolls so the
 * currently-spoken bubble stays comfortably on screen; when the page's audio
 * ends it advances to the next page.
 *
 * Design (2026-07-25 rewrite): the old model Ken-Burns *zoomed and panned* to
 * each detected box — a hard cut per line, so you never saw the artwork and the
 * camera jumped all over the page (especially on noisy OCR: art / SFX / foreign
 * script detected as "text"). This version:
 *   - never zooms — the page shows at fit-width, so the whole panel is visible;
 *   - scrolls gently, velocity-CLAMPED, so a mis-ordered/mis-detected box makes
 *     it glide, never snap;
 *   - highlights the active bubble ONLY when its box passes a sanity check
 *     (in-bounds, plausible size/aspect) — junk boxes get no highlight and no
 *     scroll target, so noise can't yank the view around.
 * (Cleaning up WHICH boxes get voiced/detected is Phase 2, in the OCR path.)
 *
 * Streaming model: each comic page is synthesized into its OWN audio artifact
 * and appended to `pages` as it finishes, so the player starts on page 1 in
 * ~20s instead of waiting for the whole chapter. While `status` is still
 * 'running', the player polls `opts.pollUrl` for newly-ready pages and, if
 * playback catches up to synthesis, shows a "synthesizing…" hold.
 *
 * MOUNTED BY NOTHING as of 2026-07-25. The in-app reader ("Listen") used to
 * mount this over itself; that took page control away from the user and gave
 * the app a SECOND page cursor that desynced from the reader's on close. It
 * now uses `narration-bar.js`, where the reader drives and audio follows.
 * This model is still the right one for a cast receiver — a TV has nobody
 * scrolling it, so audio has to drive the camera — but no cast surface calls
 * it yet (`ui/cast-comic/` renders pages only). Kept for that.
 *
 * payload (from GET/POST /api/comic-narration/{ref}[/cast]):
 *   { pages:[{ page, audio_url, duration_ms,
 *              lines:[{order,kind,text,bbox:[x,y,w,h],
 *                      audio_start_ms,audio_end_ms}] }],
 *     page_url_template, total_pages, reading_direction, status }
 * opts: { pollUrl?:string, pollMs?:number }
 *
 * bbox is normalized 0..1 (origin top-left) or null (no highlight/hold).
 * audio_start/end_ms are relative to THAT page's audio.
 */

import { createNarrationClock } from './narration-clock.js';

const _NS = 'augmentum-comic-narration';

// Max auto-scroll speed, px/frame (~540px/s @60fps). Clamping the velocity is
// what turns a mis-ordered box into a smooth glide instead of a snap.
const _MAX_SCROLL_PX_PER_FRAME = 9;
// Where the active bubble should sit in the stage: this fraction from the top.
const _COMFORT_BAND = 0.28;

function _pageUrl(template, page1) {
  // template carries the literal "{page}" token (1-indexed).
  return (template || '').replace('{page}', String(page1));
}

// Is this a plausible dialogue/narration box, or OCR noise (art / texture /
// stray glyph)? Rejects out-of-bounds, specks, near-whole-page boxes, and
// hairline slivers. A rejected box → no highlight, no scroll target.
function _bboxSane(bbox) {
  if (!Array.isArray(bbox) || bbox.length < 4) return false;
  const [x, y, w, h] = bbox;
  if (![x, y, w, h].every((n) => typeof n === 'number' && Number.isFinite(n))) return false;
  if (x < 0 || y < 0 || w <= 0 || h <= 0) return false;
  if (x + w > 1.02 || y + h > 1.02) return false;   // spills off the page
  const area = w * h;
  if (area < 0.0008 || area > 0.45) return false;    // speck or whole-panel
  const aspect = w / h;
  if (aspect > 14 || aspect < 0.06) return false;    // hairline sliver
  return true;
}

export function mountNarrationPlayer(container, payload, opts = {}) {
  if (!container || !payload) return null;
  if (!Array.isArray(payload.pages) || !payload.pages.length) return null;

  // Scheduling + streaming + the <audio> element live in the shared clock;
  // this surface owns only rendering (page image, bubble highlight, scroll).
  const template = payload.page_url_template || '';
  const pollUrl = opts.pollUrl || '';
  const pollMs = opts.pollMs || 2500;

  container.classList.add(`${_NS}-root`);
  container.innerHTML = '';

  // Stage is now a vertical SCROLL container. The page image sits at fit-width
  // in normal flow (so a tall page overflows and becomes scrollable); the
  // highlight lives inside the page wrap so it scrolls with the art.
  const stage = document.createElement('div');
  stage.className = `${_NS}-stage`;
  const pageWrap = document.createElement('div');
  pageWrap.className = `${_NS}-pagewrap`;
  const img = document.createElement('img');
  img.className = `${_NS}-page`;
  img.alt = '';
  img.decoding = 'async';
  const overlay = document.createElement('div');
  overlay.className = `${_NS}-highlight`;
  pageWrap.appendChild(img);
  pageWrap.appendChild(overlay);
  stage.appendChild(pageWrap);

  // Floating "synthesizing…" hold — child of the root (not the scroller) so it
  // stays pinned while the page scrolls underneath.
  const buffering = document.createElement('div');
  buffering.className = `${_NS}-buffering`;
  buffering.style.display = 'none';
  buffering.textContent = 'Synthesizing next page…';

  const caption = document.createElement('div');
  caption.className = `${_NS}-caption`;

  // Transport bar: play/pause + page counter. Native <audio> controls are
  // hidden because they'd only scrub WITHIN the current page's clip, which
  // reads as broken across a chained, growing track.
  const bar = document.createElement('div');
  bar.className = `${_NS}-bar`;
  const playBtn = document.createElement('button');
  playBtn.className = `${_NS}-play`;
  playBtn.type = 'button';
  playBtn.textContent = '⏸';
  const counter = document.createElement('span');
  counter.className = `${_NS}-counter`;
  bar.appendChild(playBtn);
  bar.appendChild(counter);

  container.appendChild(stage);
  container.appendChild(caption);
  container.appendChild(bar);
  container.appendChild(buffering);

  let destroyed = false;
  let _curLine = -1;      // active line index within the current page (for reflow)

  // Velocity-clamped scroll tween state.
  let _scrollTarget = 0;
  let _scrollRaf = 0;

  function showPage(page0) {
    img.src = _pageUrl(template, page0 + 1);
  }

  // Pixel rect of a normalized bbox within the rendered (fit-width) image.
  function _imgRect(bbox) {
    const iw = img.clientWidth;
    const ih = img.clientHeight;
    if (!iw || !ih) return null;
    const [x, y, w, h] = bbox;
    return { left: x * iw, top: y * ih, width: w * iw, height: h * ih };
  }

  // Position (or hide) the highlight for a bubble. Returns its rect, or null
  // when the box is missing / insane / the image isn't measurable yet.
  function highlightBubble(bubble) {
    const bbox = bubble && bubble.bbox;
    if (!bbox || !_bboxSane(bbox)) { overlay.style.opacity = '0'; return null; }
    const r = _imgRect(bbox);
    if (!r) { overlay.style.opacity = '0'; return null; }
    overlay.style.left = `${r.left.toFixed(1)}px`;
    overlay.style.top = `${r.top.toFixed(1)}px`;
    overlay.style.width = `${r.width.toFixed(1)}px`;
    overlay.style.height = `${r.height.toFixed(1)}px`;
    overlay.style.opacity = '1';
    return r;
  }

  // Ease stage.scrollTop toward _scrollTarget, capped at _MAX_SCROLL_PX_PER_FRAME.
  function _tickScroll() {
    _scrollRaf = 0;
    if (destroyed) return;
    const cur = stage.scrollTop;
    const diff = _scrollTarget - cur;
    if (Math.abs(diff) < 0.5) { stage.scrollTop = _scrollTarget; return; }
    const step = Math.sign(diff) * Math.min(Math.abs(diff) * 0.14, _MAX_SCROLL_PX_PER_FRAME);
    stage.scrollTop = cur + step;
    _scrollRaf = requestAnimationFrame(_tickScroll);
  }
  function _scrollTo(top) {
    const max = Math.max(0, stage.scrollHeight - stage.clientHeight);
    _scrollTarget = Math.max(0, Math.min(max, top));
    if (!_scrollRaf) _scrollRaf = requestAnimationFrame(_tickScroll);
  }

  // Render the active line the clock reports: caption text + highlight +
  // comfort-band scroll. index -1 clears.
  let _curLineObj = null;
  function renderLine({ index, line }) {
    _curLine = index;
    _curLineObj = line;
    if (index < 0 || !line) { caption.textContent = ''; overlay.style.opacity = '0'; return; }
    caption.textContent = line.text || '';
    caption.dataset.kind = line.kind || 'speech';
    const r = highlightBubble(line);
    // Keep the spoken bubble comfortably in view — its top ~28% down the stage.
    if (r) _scrollTo(r.top - stage.clientHeight * _COMFORT_BAND);
  }

  // Re-place the highlight (and re-aim the scroll) once the page image has real
  // dimensions, or when the viewport resizes — bbox is normalized, so pixel
  // positions must be recomputed against the current render size.
  function _reflow() {
    if (_curLine < 0 || !_curLineObj) return;
    const r = highlightBubble(_curLineObj);
    if (r) _scrollTo(r.top - stage.clientHeight * _COMFORT_BAND);
  }
  img.addEventListener('load', _reflow);
  window.addEventListener('resize', _reflow);

  // The shared clock owns scheduling + the <audio> element. This surface
  // renders what it reports.
  const clock = createNarrationClock(payload, {
    pollUrl,
    pollMs,
    onPage: ({ page }) => {
      buffering.style.display = 'none';
      // Fresh page → back to the top, highlight cleared.
      stage.scrollTop = 0;
      _scrollTarget = 0;
      overlay.style.opacity = '0';
      _curLine = -1;
      _curLineObj = null;
      showPage(page || 0);
    },
    onLine: renderLine,
    onWaiting: ({ status: st }) => {
      buffering.style.display = '';
      buffering.textContent = st === 'failed' ? 'Narration failed.' : 'Synthesizing next page…';
    },
    onFinish: () => {
      buffering.style.display = '';
      buffering.textContent = 'End of narration.';
      playBtn.textContent = '↻';
    },
    onCounter: ({ humanPage, total, status: st }) => {
      const tot = total || (st === 'done' ? '?' : '…');
      counter.textContent = `Page ${humanPage} / ${tot}`;
    },
  });
  const audio = clock.audio;
  audio.className = `${_NS}-audio`;
  container.appendChild(audio);

  // ---- transport wiring --------------------------------------------------
  playBtn.addEventListener('click', () => {
    if (playBtn.textContent === '↻') { clock.play(); playBtn.textContent = '⏸'; return; }
    if (audio.paused) { clock.resume(); playBtn.textContent = '⏸'; }
    else { clock.pause(); playBtn.textContent = '▶'; }
  });
  audio.addEventListener('play', () => { if (playBtn.textContent !== '↻') playBtn.textContent = '⏸'; });
  audio.addEventListener('pause', () => { if (!audio.ended && playBtn.textContent !== '↻') playBtn.textContent = '▶'; });

  return {
    el: container,
    play: () => clock.play(),
    pause: () => clock.pause(),
    destroy: () => {
      destroyed = true;
      if (_scrollRaf) { cancelAnimationFrame(_scrollRaf); _scrollRaf = 0; }
      clock.destroy();
      img.removeEventListener('load', _reflow);
      window.removeEventListener('resize', _reflow);
      container.innerHTML = '';
      container.classList.remove(`${_NS}-root`);
    },
  };
}

export default { mountNarrationPlayer };
