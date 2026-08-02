/**
 * Media Panel — multi-platform video player with tabs (Discover | Library | Queue).
 *
 * Evolved from YouTube Video Panel. Retains full YouTube playback, transcript
 * sync, related videos, and summary features. Adds tab navigation, platform
 * filter chips, library history, and queue management via MediaCards.
 *
 * Exports:
 *   openFromSearch(videoId, title, channel) — open panel from discovery card click
 *   openDirect(metadata) — open panel from direct URL tool result
 *   close() — close panel
 */

import { loadYouTubeAPI } from './yt-api.js';
import { FloatingVideo } from './floating-video.js';
import { renderMarkdown, highlightCodeDeferred } from './chat/markdown.js';
import { makeStreamRenderer } from './chat/stream-render.js';

const $ = (id) => document.getElementById(id);

// State
let _player = null;
let _videoId = null;
let _paragraphs = [];
let _syncInterval = null;
let _autoScroll = false;  // Off by default — user opts in via "Follow along" pill
let _emitted30s = false;
let _activeTab = 'discover';
let _activePlatformFilter = 'all';
let _lastSearchResults = [];

/** Emit a discovery signal to the backend. Fire-and-forget. */
function _emitSignal(signalType, data = {}) {
  if (!window.appSettings?.discoveryEnabled) return;
  const body = { signal_type: signalType, ...data };
  fetch('/api/discovery/signal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch(() => {});
}

// DOM refs (resolved lazily)
const _dom = {};
function _resolveDom() {
  _dom.panel = $('youtube-panel');
  _dom.title = $('yt-panel-title');
  _dom.body = $('yt-panel-body');
  _dom.playerWrap = $('yt-player-wrap');
  _dom.meta = $('yt-video-meta');
  _dom.actions = $('yt-actions');
  _dom.summarize = $('yt-summarize');
  _dom.addToPlaylist = $('yt-add-to-playlist');
  _dom.original = $('yt-original');
  _dom.summary = $('yt-summary');
  _dom.transcript = $('yt-transcript');
  _dom.resumePill = $('yt-resume-pill');

  // Wire once — reads current video state from module vars at click time so
  // we don't need to re-bind on every swap like the summarize button.
  if (_dom.addToPlaylist && !_dom.addToPlaylist.dataset.bound) {
    _dom.addToPlaylist.dataset.bound = '1';
    _dom.addToPlaylist.addEventListener('click', () => {
      if (!_videoId) return;
      window.dispatchEvent(new CustomEvent('playlist:add-item', {
        detail: {
          type: 'youtube',
          videoId: _videoId,
          title: _dom.title?.textContent || '',
          channel: _dom.meta?.textContent || '',
          thumbnail: `https://i.ytimg.com/vi/${_videoId}/mqdefault.jpg`,
        },
      }));
    });
  }
}

// ---- Player ----

async function _createPlayer(videoId) {
  await loadYouTubeAPI();

  // Destroy previous player if any
  if (_player && _player.destroy) {
    try { _player.destroy(); } catch {}
  }

  // Clear the container and create fresh target div
  _dom.playerWrap.innerHTML = '<div id="yt-player"></div>';

  return new Promise((resolve) => {
    _player = new YT.Player('yt-player', {
      videoId,
      playerVars: {
        autoplay: 0,
        modestbranding: 1,
        rel: 0,              // minimize related videos
        cc_load_policy: 0,   // we provide our own transcript
        playsinline: 1,
        disablekb: 0,
        iv_load_policy: 3,   // disable annotations
      },
      events: {
        onReady: () => resolve(_player),
        onStateChange: _onPlayerStateChange,
      },
    });
  });
}

function _onPlayerStateChange(event) {
  // YT.PlayerState: PLAYING=1, PAUSED=2, ENDED=0, BUFFERING=3
  if (event.data === 1) {
    _startSync();
    _hideEndOverlay();
  } else if (event.data === 0) {
    // Video ended — show our own overlay so YouTube's related videos are covered
    _stopSync();
    _showEndOverlay();
  } else {
    _stopSync();
  }
}

function _showEndOverlay() {
  if (!_dom.playerWrap) return;
  let overlay = _dom.playerWrap.querySelector('.yt-end-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.className = 'yt-end-overlay';
    overlay.innerHTML =
      '<button class="yt-end-btn yt-end-replay">Replay</button>' +
      '<button class="yt-end-btn yt-end-back">Back to results</button>';
    overlay.querySelector('.yt-end-replay').addEventListener('click', () => {
      if (_player && _player.seekTo) { _player.seekTo(0); _player.playVideo(); }
    });
    overlay.querySelector('.yt-end-back').addEventListener('click', () => {
      close();
    });
    _dom.playerWrap.appendChild(overlay);
  }
  overlay.classList.remove('hidden');
}

function _hideEndOverlay() {
  if (!_dom.playerWrap) return;
  const overlay = _dom.playerWrap.querySelector('.yt-end-overlay');
  if (overlay) overlay.classList.add('hidden');
}

// ---- Transcript Sync ----

function _startSync() {
  if (_syncInterval) return;
  _syncInterval = setInterval(() => {
    if (!_player || !_player.getCurrentTime) return;
    const time = _player.getCurrentTime();
    _highlightParagraph(time);
    if (_autoScroll) _scrollToActive();
    if (!_emitted30s && time >= 30) {
      _emitted30s = true;
      const duration = _player.getDuration?.() || 0;
      _emitSignal('video_watch', {
        source_url: `https://www.youtube.com/watch?v=${_videoId}`,
        source_title: _dom?.title?.textContent || '',
        content_type: 'video',
        source_type: 'video_transcript',
        weight: 1.5,
        metadata: { video_id: _videoId, progress_seconds: Math.round(time), total_duration: Math.round(duration) },
        raw_content: _paragraphs?.map(p => p.text).join(' ') || '',
        is_html: false,
      });
    }
  }, 500);
}

function _stopSync() {
  if (_syncInterval) {
    clearInterval(_syncInterval);
    _syncInterval = null;
  }
}

function _highlightParagraph(time) {
  const paras = _dom.transcript.querySelectorAll('.yt-paragraph');
  let activeIdx = -1;

  for (let i = 0; i < _paragraphs.length; i++) {
    if (_paragraphs[i].start <= time) {
      activeIdx = i;
    } else {
      break;
    }
  }

  paras.forEach((el, i) => {
    el.classList.toggle('yt-line-active', i === activeIdx);
  });
}

/** Return the scroll container — discover tab when tabs injected, otherwise body. */
function _scrollContainer() {
  return $('yt-tab-discover') || _dom.body;
}

function _scrollToActive() {
  const active = _dom.transcript.querySelector('.yt-line-active');
  const container = _scrollContainer();
  if (!active || !container) return;
  // Only scroll if the active line is out of the visible area
  const containerRect = container.getBoundingClientRect();
  const activeRect = active.getBoundingClientRect();
  const isVisible = activeRect.top >= containerRect.top && activeRect.bottom <= containerRect.bottom;
  if (!isVisible) {
    // Scroll so active line is ~30% from top of panel
    const targetTop = active.offsetTop - container.offsetTop - (containerRect.height * 0.3);
    container.scrollTop = targetTop;
  }
}

// ---- Transcript Rendering ----

function _renderTranscript(paragraphs) {
  _paragraphs = paragraphs;

  if (!paragraphs || paragraphs.length === 0) {
    _dom.transcript.innerHTML = `
      <div class="yt-no-transcript">
        <p>No transcript available for this video.</p>
        <p class="yt-no-transcript-hint">Captions may be disabled by the creator.</p>
      </div>`;
    _dom.summarize.style.display = 'none';
    return;
  }

  _dom.summarize.style.display = '';
  let html = '';
  for (const p of paragraphs) {
    const ts = _formatTimestamp(p.start);
    const escaped = _esc(p.text);
    html += `<div class="yt-paragraph" data-start="${p.start}">
      <span class="yt-timestamp" data-start="${p.start}">${ts}</span>
      <span class="yt-text">${escaped}</span>
    </div>`;
  }
  _dom.transcript.innerHTML = html;
}

function _formatTimestamp(seconds) {
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function _esc(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

// ---- Manual Scroll Detection ----

let _userScrollTimeout = null;

function _initScrollDetection() {
  const el = _scrollContainer();

  // When user scrolls while auto-follow is on, pause it
  el.addEventListener('scroll', () => {
    if (_autoScroll) {
      _autoScroll = false;
      _dom.resumePill.textContent = 'Resume following';
      _dom.resumePill.classList.remove('hidden');
    }
    clearTimeout(_userScrollTimeout);
    _userScrollTimeout = setTimeout(() => {}, 1000);
  }, { passive: true });

  // Pill toggles auto-follow on
  _dom.resumePill.addEventListener('click', () => {
    _autoScroll = true;
    _dom.resumePill.classList.add('hidden');
    _scrollToActive();
  });
}

// ---- Click to Seek ----

function _initClickToSeek() {
  _dom.transcript.addEventListener('click', (e) => {
    const para = e.target.closest('.yt-paragraph');
    if (para && _player && _player.seekTo) {
      const start = parseFloat(para.dataset.start);
      _player.seekTo(start, true);
      _autoScroll = true;
      _dom.resumePill.classList.add('hidden');
      _emitSignal('video_seek', {
        source_url: `https://www.youtube.com/watch?v=${_videoId}`,
        source_title: _dom?.title?.textContent || '',
        content_type: 'video',
        weight: 0.5,
        metadata: { video_id: _videoId, timestamp: start },
      });
    }
  });
}

// ---- Mini-Player (Desktop) ----

let _miniObserver = null;

function _initMiniPlayer() {
  // Mini-player disabled — the sticky float + IntersectionObserver combo
  // causes scroll-fighting on many browsers. The player stays at the top
  // of the panel, and the transcript scrolls below it naturally.
  // Tracked in docs/research-backlog.md (YouTube panel — desktop mini-player).
}

// ---- Summarize ----

const _SUMMARY_PROMPT = `Summarize this YouTube video transcript. Structure your summary as:

**Key Points**
- 3-5 bullet points covering the main ideas, each referencing a timestamp [MM:SS]

**Overview**
A 2-3 sentence description of what the video covers and the creator's main argument or conclusion.

Be concise. Don't repeat the title. Reference timestamps so the viewer can jump to sections that interest them.`;

async function _summarize(title) {
  _emitSignal('video_summary', {
    source_url: `https://www.youtube.com/watch?v=${_videoId}`,
    source_title: title,
    content_type: 'video',
    weight: 2.5,
    metadata: { video_id: _videoId },
  });

  const fullText = _paragraphs.map(p => `[${_formatTimestamp(p.start)}] ${p.text}`).join('\n');
  const truncated = fullText.length > 15000
    ? fullText.slice(0, 15000) + '\n\n[Transcript truncated]'
    : fullText;

  _dom.summary.classList.remove('hidden');
  _dom.summary.innerHTML = '<div class="yt-summary-loading">Summarizing...</div>';
  _dom.summarize.disabled = true;
  _dom.summarize.textContent = 'Summarizing...';

  try {
    // Get current model from app state
    const appModule = await import('./app.js');
    const model = appModule.default?.state?.currentModel || '';

    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Augmentum-Mode': 'passthrough',
        'X-Augmentum-Tools': 'none',
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: _SUMMARY_PROMPT },
          { role: 'user', content: `Video: "${title}"\n\nTranscript:\n${truncated}` },
        ],
        stream: true,
      }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    // Stream the response — handle both OpenAI SSE and Ollama NDJSON formats
    _dom.summary.innerHTML = '';
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullContent = '';

    // Shared incremental renderer (chat/stream-render.js) — coalesced + split
    // so the summary doesn't re-parse from scratch on every network chunk.
    // compact = full chat markdown minus the chat-only code toolbar.
    const aiRender = makeStreamRenderer(_dom.summary, { compact: true });

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        // OpenAI SSE format: data: {"choices":[{"delta":{"content":"..."}}]}
        if (trimmed.startsWith('data: ')) {
          const data = trimmed.slice(6).trim();
          if (data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta?.content
                       || parsed.message?.content
                       || '';
            if (delta) fullContent += delta;
          } catch { /* ignore */ }
          continue;
        }

        // Ollama NDJSON format: {"message":{"content":"..."}}
        if (trimmed.startsWith('{')) {
          try {
            const parsed = JSON.parse(trimmed);
            const delta = parsed.message?.content
                       || parsed.choices?.[0]?.delta?.content
                       || '';
            if (delta) fullContent += delta;
          } catch { /* ignore */ }
        }
      }

      aiRender.render(fullContent);
    }

    // Flatten the streaming split into one render + highlight off the critical path.
    _dom.summary.innerHTML = renderMarkdown(fullContent, { compact: true });
    highlightCodeDeferred(_dom.summary);
    _dom.summarize.textContent = 'Summary \u2713';
    _dom.summarize.classList.add('yt-summarized');

  } catch (err) {
    _dom.summary.innerHTML = `<div class="yt-summary-loading">Summary failed: ${_esc(err.message)}</div>`;
    _dom.summarize.disabled = false;
    _dom.summarize.textContent = 'Retry Summary';
  }
}


// ---- Related Videos (rabbit-hole browsing) ----

async function _loadRelatedVideos(title, currentVideoId) {
  const relatedEl = $('yt-related-content');
  const toggleBtn = $('yt-related-toggle');
  const section = $('yt-related-section');
  if (!relatedEl || !section) return;

  // Wire toggle (once)
  if (toggleBtn && !toggleBtn._wired) {
    toggleBtn._wired = true;
    toggleBtn.addEventListener('click', () => {
      const content = $('yt-related-content');
      const isHidden = content.classList.contains('hidden');
      content.classList.toggle('hidden');
      toggleBtn.classList.toggle('yt-related-open', isHidden);
    });
  }

  relatedEl.innerHTML = '<div class="yt-related-loading">Finding related videos...</div>';

  try {
    // Search for related videos using the video title as query (first 6 words)
    const query = title.replace(/[^\w\s]/g, '').trim().split(/\s+/).slice(0, 6).join(' ');
    const searchResp = await fetch(`/api/youtube/related?q=${encodeURIComponent(query)}&exclude=${currentVideoId}`);
    if (!searchResp.ok) {
      relatedEl.innerHTML = '';
      return;
    }
    const data = await searchResp.json();
    if (!data.results || data.results.length === 0) {
      relatedEl.innerHTML = '';
      return;
    }

    let html = '<div class="yt-related-grid">';
    for (const r of data.results) {
      const thumbUrl = `/api/browse/image?url=${encodeURIComponent(r.thumbnail)}`;
      html += `<div class="yt-related-card" data-video-id="${_esc(r.video_id)}" data-title="${_esc(r.title)}" data-channel="${_esc(r.channel)}">
        <div class="yt-related-thumb">
          <img src="${_esc(thumbUrl)}" alt="${_esc(r.title)}" loading="lazy" onerror="this.style.display='none'">
          ${r.duration ? `<span class="yt-card-duration">${_esc(String(r.duration))}</span>` : ''}
        </div>
        <div class="yt-related-info">
          <div class="yt-related-title">${_esc(r.title)}</div>
          <div class="yt-related-channel">${_esc(r.channel)}</div>
        </div>
      </div>`;
    }
    html += '</div>';
    relatedEl.innerHTML = html;

    // Click to swap video in-place
    relatedEl.addEventListener('click', (e) => {
      const card = e.target.closest('.yt-related-card');
      if (!card) return;
      _swapVideo(card.dataset.videoId, card.dataset.title, card.dataset.channel);
    });

  } catch {
    relatedEl.innerHTML = '';
  }
}

async function _swapVideo(videoId, title, channel) {
  _videoId = videoId;
  _stopSync();
  _hideEndOverlay();

  // Update panel header
  _dom.title.textContent = title;
  _dom.meta.textContent = channel;
  _dom.original.href = `https://www.youtube.com/watch?v=${videoId}`;

  // Reset summary
  _dom.summary.classList.add('hidden');
  _dom.summary.innerHTML = '';
  _dom.summarize.disabled = false;
  _dom.summarize.textContent = 'Summarize';
  _dom.summarize.classList.remove('yt-summarized');
  _dom.summarize.style.display = '';
  _dom.summarize.onclick = () => _summarize(title);

  // Swap player
  await _createPlayer(videoId);

  _emitted30s = false;  // Reset for new video
  _emitSignal('video_open', {
    source_url: `https://www.youtube.com/watch?v=${videoId}`,
    source_title: title,
    content_type: 'video',
    weight: 1.0,
    metadata: { video_id: videoId, channel },
  });

  // Fetch new transcript
  _dom.transcript.innerHTML = '<div class="yt-summary-loading">Loading transcript...</div>';
  try {
    const resp = await fetch(`/api/youtube/transcript?v=${videoId}`);
    const data = await resp.json();
    if (data.title) _dom.title.textContent = data.title;
    if (data.channel) _dom.meta.textContent = data.channel;
    _renderTranscript(data.paragraphs || []);
    if (data.transcript_error) {
      _dom.summarize.style.display = 'none';
    }
  } catch {
    _dom.transcript.innerHTML = '<div class="yt-no-transcript"><p>Failed to load transcript.</p></div>';
    _dom.summarize.style.display = 'none';
  }

  // Load new related videos
  _loadRelatedVideos(title, videoId);

  // Scroll to top of panel
  const sc = _scrollContainer();
  if (sc) sc.scrollTop = 0;
}

// ---- Tab System ----

let _tabsInjected = false;

function _injectTabs() {
  if (_tabsInjected) return;
  _tabsInjected = true;

  // Inject tab pills after the header
  const header = _dom.panel.querySelector('.yt-panel-header');
  if (!header) return;

  const tabBar = document.createElement('div');
  tabBar.className = 'yt-panel-tabs';
  tabBar.id = 'yt-panel-tabs';
  tabBar.innerHTML = `
    <button class="yt-panel-tab active" data-tab="discover">Discover</button>
    <button class="yt-panel-tab" data-tab="library">Library</button>
    <button class="yt-panel-tab" data-tab="queue">Queue</button>
  `;
  header.insertAdjacentElement('afterend', tabBar);

  // Wrap existing body content in a discover tab wrapper
  const body = _dom.body;
  const existingChildren = Array.from(body.children);
  const discoverWrap = document.createElement('div');
  discoverWrap.className = 'yt-tab-content active';
  discoverWrap.id = 'yt-tab-discover';
  for (const child of existingChildren) {
    discoverWrap.appendChild(child);
  }
  body.appendChild(discoverWrap);

  // Create library tab content
  const libraryWrap = document.createElement('div');
  libraryWrap.className = 'yt-tab-content';
  libraryWrap.id = 'yt-tab-library';
  libraryWrap.innerHTML = '<div class="yt-queue-empty">Loading history...</div>';
  body.appendChild(libraryWrap);

  // Create queue tab content
  const queueWrap = document.createElement('div');
  queueWrap.className = 'yt-tab-content';
  queueWrap.id = 'yt-tab-queue';
  queueWrap.innerHTML = '<div class="yt-queue-empty">Your queue is empty. Add videos with the queue button on any card.</div>';
  body.appendChild(queueWrap);

  // Wire tab clicks
  tabBar.addEventListener('click', (e) => {
    const btn = e.target.closest('.yt-panel-tab');
    if (!btn) return;
    _switchTab(btn.dataset.tab);
  });

  // Listen for queue updates to re-render
  window.addEventListener('media:queue-updated', () => {
    if (_activeTab === 'queue') _renderQueue();
  });
}

function _switchTab(tab) {
  _activeTab = tab;

  // Update tab buttons
  const tabs = _dom.panel.querySelectorAll('.yt-panel-tab');
  tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tab));

  // Update tab content
  const contents = _dom.body.querySelectorAll('.yt-tab-content');
  contents.forEach(c => c.classList.toggle('active', c.id === `yt-tab-${tab}`));

  // Lazy load content
  if (tab === 'library') _loadLibrary();
  if (tab === 'queue') _renderQueue();
}

// ---- Platform Filter Chips ----

let _filtersInjected = false;

function _injectPlatformFilters() {
  if (_filtersInjected) return;
  _filtersInjected = true;

  const discoverTab = $('yt-tab-discover');
  if (!discoverTab) return;

  const filtersEl = document.createElement('div');
  filtersEl.className = 'yt-platform-filters';
  filtersEl.id = 'yt-platform-filters';

  const MC = window.MediaCards;
  const platforms = MC ? Object.keys(MC.PLATFORMS) : [];
  const chips = [{ key: 'all', label: 'All' }];
  for (const p of platforms) {
    if (p === 'unknown') continue;
    chips.push({ key: p, label: MC.PLATFORMS[p].label });
  }

  filtersEl.innerHTML = chips.map(c =>
    `<button class="yt-platform-chip${c.key === 'all' ? ' active' : ''}" data-platform="${_esc(c.key)}">${_esc(c.label)}</button>`
  ).join('');

  // Insert at the top of discover tab (before the player)
  discoverTab.insertBefore(filtersEl, discoverTab.firstChild);

  filtersEl.addEventListener('click', (e) => {
    const chip = e.target.closest('.yt-platform-chip');
    if (!chip) return;
    _activePlatformFilter = chip.dataset.platform;
    filtersEl.querySelectorAll('.yt-platform-chip').forEach(c =>
      c.classList.toggle('active', c.dataset.platform === _activePlatformFilter)
    );
    _applyPlatformFilter();
  });
}

function _applyPlatformFilter() {
  // Filter related videos section by platform (if MediaCards is available)
  const MC = window.MediaCards;
  if (!MC) return;

  const relatedCards = _dom.panel.querySelectorAll('#yt-tab-discover .media-card');
  relatedCards.forEach(card => {
    const platform = card.dataset.platform || 'unknown';
    if (_activePlatformFilter === 'all' || platform === _activePlatformFilter) {
      card.style.display = '';
    } else {
      card.style.display = 'none';
    }
  });

  // Also filter standard related cards
  const standardCards = _dom.panel.querySelectorAll('#yt-tab-discover .yt-related-card');
  if (_activePlatformFilter === 'all' || _activePlatformFilter === 'youtube') {
    standardCards.forEach(c => c.style.display = '');
  } else {
    standardCards.forEach(c => c.style.display = 'none');
  }
}

// ---- Library Tab ----

let _libraryLoaded = false;

async function _loadLibrary() {
  const wrap = $('yt-tab-library');
  if (!wrap) return;

  // Reload each time tab is visited (fresh data)
  wrap.innerHTML = '<div class="yt-queue-empty">Loading history...</div>';

  try {
    const resp = await fetch('/api/discovery/history?days=30');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const items = (data.items || []).filter(i => i.content_type === 'video');

    if (items.length === 0) {
      wrap.innerHTML = '<div class="yt-queue-empty">No video history yet. Videos you watch will appear here.</div>';
      return;
    }

    // Group by day
    const groups = {};
    for (const item of items) {
      const date = item.last_visited || item.created_at || '';
      const day = date.slice(0, 10) || 'Unknown';
      if (!groups[day]) groups[day] = [];
      groups[day].push(item);
    }

    const MC = window.MediaCards;
    let html = '';
    for (const [day, entries] of Object.entries(groups)) {
      const label = _formatDayLabel(day);
      html += `<div class="yt-library-day">${_esc(label)}</div>`;
      for (const entry of entries) {
        if (MC) {
          html += MC.renderCard({
            title: entry.title || 'Untitled',
            url: entry.url || '',
            thumbnail: entry.thumbnail || '',
            channel: entry.domain || '',
          }, { hideActions: false });
        } else {
          html += `<div class="yt-related-card" style="padding:8px;cursor:pointer" data-url="${_esc(entry.url || '')}">
            <div class="yt-related-info">
              <div class="yt-related-title">${_esc(entry.title || 'Untitled')}</div>
              <div class="yt-related-channel">${_esc(entry.domain || '')}</div>
            </div>
          </div>`;
        }
      }
    }

    wrap.innerHTML = html;

    if (MC) {
      MC.wireActions(wrap);
    }

    // Also handle non-MediaCard clicks
    wrap.addEventListener('click', (e) => {
      const card = e.target.closest('.yt-related-card[data-url]');
      if (!card) return;
      const url = card.dataset.url;
      if (url) window.open(url, '_blank');
    });

    _libraryLoaded = true;
  } catch (err) {
    wrap.innerHTML = `<div class="yt-queue-empty">Failed to load history: ${_esc(err.message)}</div>`;
  }
}

function _formatDayLabel(dateStr) {
  if (!dateStr || dateStr === 'Unknown') return 'Unknown';
  try {
    const d = new Date(dateStr);
    const now = new Date();
    const diffMs = now - d;
    const diffDays = Math.floor(diffMs / 86400000);
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return d.toLocaleDateString('en-US', { weekday: 'long' });
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return dateStr;
  }
}

// ---- Queue Tab ----

function _renderQueue() {
  const wrap = $('yt-tab-queue');
  if (!wrap) return;

  const MC = window.MediaCards;
  const queue = MC ? MC.getQueue() : [];

  if (queue.length === 0) {
    wrap.innerHTML = '<div class="yt-queue-empty">Your queue is empty. Add videos with the queue button on any card.</div>';
    return;
  }

  let html = '';
  for (let i = 0; i < queue.length; i++) {
    const video = queue[i];
    if (MC) {
      // Wrap each card with a position-relative container for the remove button
      html += `<div style="position:relative">
        ${MC.renderCard(video, { hideActions: true })}
        <button class="yt-queue-remove" data-index="${i}" title="Remove from queue">&times;</button>
      </div>`;
    } else {
      html += `<div class="yt-related-card" style="padding:8px">
        <div class="yt-related-info">
          <div class="yt-related-title">${_esc(video.title || 'Untitled')}</div>
        </div>
      </div>`;
    }
  }

  wrap.innerHTML = html;

  if (MC) MC.wireActions(wrap);

  // Wire remove buttons
  wrap.querySelectorAll('.yt-queue-remove').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.index, 10);
      if (MC) MC.removeFromQueue(idx);
      // queue-updated event will trigger re-render
    });
  });
}

// ---- Panel Title Update ----

function _updatePanelHeaderLabel() {
  // Update the panel back button tooltip to say "Media" instead of YouTube
  const backBtn = $('yt-panel-back');
  if (backBtn) backBtn.title = 'Back';
}

// ---- Public API ----

export async function openFromSearch(videoId, title, channel) {
  _resolveDom();
  _wireButtons();
  _injectTabs();
  _injectPlatformFilters();
  _updatePanelHeaderLabel();
  _videoId = videoId;

  // Show panel and switch to discover tab
  _dom.panel.classList.remove('hidden');
  _switchTab('discover');
  _dom.title.textContent = title;
  _dom.meta.textContent = channel;
  _dom.summary.classList.add('hidden');
  _dom.summary.innerHTML = '';
  _dom.summarize.disabled = false;
  _dom.summarize.textContent = 'Summarize';
  _dom.summarize.classList.remove('yt-summarized');
  _dom.original.href = `https://www.youtube.com/watch?v=${videoId}`;
  _dom.transcript.innerHTML = '<div class="yt-summary-loading">Loading transcript...</div>';

  // Create player
  await _createPlayer(videoId);

  _emitted30s = false;  // Reset for new video
  _emitSignal('video_open', {
    source_url: `https://www.youtube.com/watch?v=${videoId}`,
    source_title: title,
    content_type: 'video',
    weight: 1.0,
    metadata: { video_id: videoId, channel },
  });

  // Fetch transcript
  try {
    const resp = await fetch(`/api/youtube/transcript?v=${videoId}`);
    const data = await resp.json();

    if (data.title) _dom.title.textContent = data.title;
    if (data.channel) {
      _dom.meta.textContent = data.channel;
    }

    _renderTranscript(data.paragraphs || []);

    if (data.transcript_error) {
      _dom.transcript.innerHTML = `
        <div class="yt-no-transcript">
          <p>No transcript available for this video.</p>
          <p class="yt-no-transcript-hint">${_esc(data.transcript_error)}</p>
        </div>`;
      _dom.summarize.style.display = 'none';
    }
  } catch (err) {
    _dom.transcript.innerHTML = `<div class="yt-no-transcript"><p>Failed to load transcript.</p></div>`;
    _dom.summarize.style.display = 'none';
  }

  // Init interactions
  _initScrollDetection();
  _initClickToSeek();
  _initMiniPlayer();

  // Wire summarize button
  _dom.summarize.onclick = () => _summarize(title);

  // Load related videos for rabbit-hole browsing
  _loadRelatedVideos(title, videoId);
}

export async function openDirect(metadata) {
  _resolveDom();
  _wireButtons();
  _injectTabs();
  _injectPlatformFilters();
  _updatePanelHeaderLabel();
  _videoId = metadata.video_id;

  _dom.panel.classList.remove('hidden');
  _switchTab('discover');
  _dom.title.textContent = metadata.title || '';
  _dom.meta.textContent = metadata.channel || '';
  _dom.summary.classList.add('hidden');
  _dom.summary.innerHTML = '';
  _dom.summarize.disabled = false;
  _dom.summarize.textContent = 'Summarize';
  _dom.summarize.classList.remove('yt-summarized');
  _dom.original.href = metadata.url || `https://www.youtube.com/watch?v=${metadata.video_id}`;

  await _createPlayer(metadata.video_id);

  _emitted30s = false;  // Reset for new video
  _emitSignal('video_open', {
    source_url: `https://www.youtube.com/watch?v=${metadata.video_id}`,
    source_title: metadata.title || '',
    content_type: 'video',
    weight: 1.0,
    metadata: { video_id: metadata.video_id, channel: metadata.channel || '' },
  });

  _renderTranscript(metadata.paragraphs || []);

  if (metadata.transcript_error) {
    _dom.summarize.style.display = 'none';
  }

  _initScrollDetection();
  _initClickToSeek();
  _initMiniPlayer();

  _dom.summarize.onclick = () => _summarize(metadata.title || '');
}

/** Hand the current video off to FloatingVideo by spinning up a fresh
 *  ``YT.Player`` inside FloatingVideo's slot at the captured progress.
 *
 *  Why not reparent the existing iframe? Chrome reloads an iframe when
 *  its parent element changes, which voids the ``YT.Player`` instance
 *  and restarts playback from 0. We sidestep that by:
 *
 *    1. Capturing currentTime + play-state off the live player.
 *    2. Asking FloatingVideo to ``mount`` — we build a new player
 *       in its slot with ``playerVars.start`` = captured time so
 *       playback resumes mid-video instead of from the beginning.
 *    3. Destroying the old player after the new one mounts.
 *
 *  The api object we hand FloatingVideo is full-surface (seek / time /
 *  duration / play / pause / mute / volume) so the floating chrome's
 *  buttons drive the new player directly. */
function _popOut() {
  if (!_player || !_videoId) return;
  const data = (typeof _player.getVideoData === 'function') ? _player.getVideoData() : {};
  const videoId = _videoId;

  // Capture playback state BEFORE we tear anything down.
  let startSec = 0;
  let wasPlaying = true;
  try {
    startSec = Math.max(0, Math.floor(Number(_player.getCurrentTime?.() || 0)));
    const state = _player.getPlayerState?.();
    // YT.PlayerState: -1 unstarted, 0 ended, 1 playing, 2 paused,
    // 3 buffering, 5 cued. Treat playing OR buffering as "should
    // resume playing".
    wasPlaying = (state === 1 || state === 3);
  } catch { /* keep defaults */ }

  const oldPlayer = _player;  // destroy after the new player mounts

  FloatingVideo.open({
    fileId: `yt:${videoId}`,
    videoId,
    title: data.title || '',
    channel: data.author || '',
    thumbnail: `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`,
    mode: 'companion',
    mount: (slot) => {
      const mountId = `fv-yt-mount-${Date.now().toString(36)}`;
      slot.innerHTML = `<div id="${mountId}"></div>`;
      const player = new YT.Player(mountId, {
        videoId,
        playerVars: {
          autoplay: wasPlaying ? 1 : 0,
          start: startSec,
          // controls: 0 hides YouTube's native chrome entirely — its
          // seek bar, play/pause, volume, time display, fullscreen,
          // and settings gear. FloatingVideo's overlay drives all of
          // those now via the api object below, so the native bar
          // would just be visual duplication. disablekb mirrors that
          // for keyboard input — FloatingVideo owns the hotkeys.
          controls: 0,
          disablekb: 1,
          modestbranding: 1,
          rel: 0,
          cc_load_policy: 0,
          playsinline: 1,
          iv_load_policy: 3,
        },
        events: {
          // Defensive re-seek in case ``playerVars.start`` was ignored
          // (some embed contexts strip it). The seek is a no-op when
          // we're already at startSec.
          onReady: () => { try { player.seekTo(startSec, true); } catch {} },
        },
      });
      return {
        iframe: slot.querySelector('iframe'),
        api: {
          seekTo: (s) => {
            try { player.seekTo(Math.max(0, Number(s) || 0), true); } catch {}
          },
          getCurrentTime: () => {
            try { return Number(player.getCurrentTime?.() || 0); } catch { return 0; }
          },
          getDuration: () => {
            try { return Number(player.getDuration?.() || 0); } catch { return 0; }
          },
          canSeek: () => true,
          togglePlay: () => {
            try {
              const s = player.getPlayerState?.();
              if (s === 1) player.pauseVideo();
              else player.playVideo();
            } catch {}
          },
          play: () => { try { player.playVideo(); } catch {} },
          pause: () => { try { player.pauseVideo(); } catch {} },
          setVolume: (v) => {
            try { player.setVolume(Math.round(Math.max(0, Math.min(1, v)) * 100)); } catch {}
          },
          getVolume: () => {
            try { return (Number(player.getVolume?.() || 0)) / 100; } catch { return 1; }
          },
          toggleMute: () => {
            try {
              if (player.isMuted?.()) player.unMute();
              else player.mute();
            } catch {}
          },
          isMuted: () => {
            try { return !!player.isMuted?.(); } catch { return false; }
          },
          isPlaying: () => {
            try {
              const s = player.getPlayerState?.();
              return s === 1 || s === 3;  // PLAYING or BUFFERING
            } catch { return false; }
          },
          destroy: () => { try { player.destroy?.(); } catch {} },
        },
      };
    },
  });

  // Old player is now redundant — destroy it so we don't leave two
  // YouTube instances burning audio. Do this AFTER FloatingVideo.open
  // returns so a synchronous mount error doesn't kill playback entirely.
  _stopSync();
  if (_miniObserver) { _miniObserver.disconnect(); _miniObserver = null; }
  try { oldPlayer.destroy?.(); } catch {}
  _player = null;
  _videoId = null;
  _paragraphs = [];
  _autoScroll = false;
  _dom.panel?.classList.add('hidden');
}

export function close() {
  _stopSync();
  if (_miniObserver) { _miniObserver.disconnect(); _miniObserver = null; }
  if (_player && _player.destroy) { try { _player.destroy(); } catch {} }
  _player = null;
  _videoId = null;
  _paragraphs = [];
  _autoScroll = false;
  _activeTab = 'discover';

  const panel = $('youtube-panel');
  if (panel) panel.classList.add('hidden');
}

// ---- Init: wire close/back/popout buttons ----
let _buttonsWired = false;
function _wireButtons() {
  if (_buttonsWired) return;
  const closeBtn = $('yt-panel-close');
  const backBtn = $('yt-panel-back');
  const popoutBtn = $('yt-panel-popout');
  if (closeBtn) { closeBtn.addEventListener('click', close); _buttonsWired = true; }
  if (backBtn) backBtn.addEventListener('click', close);
  if (popoutBtn) popoutBtn.addEventListener('click', _popOut);

  const transcriptToggle = $('yt-transcript-toggle');
  const transcriptEl = $('yt-transcript');
  if (transcriptToggle && transcriptEl && !transcriptToggle._wired) {
    transcriptToggle._wired = true;
    transcriptToggle.addEventListener('click', () => {
      const isHidden = transcriptEl.classList.toggle('hidden');
      transcriptToggle.classList.toggle('yt-transcript-open', !isHidden);
    });
  }
}
// Try immediately (DOM likely ready since we're imported dynamically)
_wireButtons();
// Fallback if somehow imported before DOM ready
if (!_buttonsWired) {
  document.addEventListener('DOMContentLoaded', _wireButtons);
}
// Expose close() on window so browse panel can stop playback when closing
window._ytPanel = { close };

// Cross-surface: receive "Play" from media cards / browse embeds
window.addEventListener('media:play', (e) => {
  const video = e.detail;
  if (!video) return;
  const videoId = video.videoId || video.video_id || '';
  if (videoId) {
    openFromSearch(videoId, video.title || '', video.channel || video.author || '');
  } else if (video.url) {
    // Non-YouTube platform — open URL in new tab for now.
    // Embed support tracked in docs/research-backlog.md.
    window.open(video.url, '_blank');
  }
});

// Allow other surfaces to open panel to a specific tab
window.addEventListener('media:open-panel', (e) => {
  const tab = e.detail?.tab || 'discover';
  _resolveDom();
  _wireButtons();
  _injectTabs();
  _injectPlatformFilters();
  _updatePanelHeaderLabel();
  _dom.panel.classList.remove('hidden');
  _switchTab(tab);
});
