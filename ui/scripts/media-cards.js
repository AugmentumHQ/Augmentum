/* media-cards.js — shared MediaCard component for Augmentum
   Platform detection, card rendering, action wiring, queue management. */
(function () {
  'use strict';

  const esc = window.escapeHtml ||
    ((s) => String(s).replace(/[&<>"`$]/g, c =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', '`': '&#96;', '$': '&#36;' }[c] || c)));

  /* ── Platform registry ────────────────────────────────────────── */

  const PLATFORMS = {
    youtube:      { label: 'YouTube',      color: '#ff4444', icon: '▶' },
    peertube:     { label: 'PeerTube',     color: '#34d399', icon: '◆' },
    archive:      { label: 'Archive',      color: '#f59e0b', icon: '●' },
    bilibili:     { label: 'Bilibili',     color: '#00a1d6', icon: '◈' },
    odysee:       { label: 'Odysee',       color: '#a78bfa', icon: '◇' },
    rumble:       { label: 'Rumble',       color: '#85c742', icon: '▶' },
    kick:         { label: 'Kick',         color: '#53fc18', icon: '▶' },
    vimeo:        { label: 'Vimeo',        color: '#1ab7ea', icon: '▶' },
    dailymotion:  { label: 'Dailymotion',  color: '#0066dc', icon: '▶' },
    nebula:       { label: 'Nebula',       color: '#5850ec', icon: '▶' },
    unknown:      { label: 'Video',        color: '#888',    icon: '▶' },
  };

  /* ── Platform detection ───────────────────────────────────────── */

  function detectPlatform(url) {
    if (!url) return 'unknown';
    const u = url.toLowerCase();
    if (u.includes('youtube.com') || u.includes('youtu.be'))       return 'youtube';
    if (u.includes('vimeo.com'))                                    return 'vimeo';
    if (u.includes('bilibili.com') || u.includes('b23.tv'))        return 'bilibili';
    if (u.includes('dailymotion.com'))                              return 'dailymotion';
    if (u.includes('rumble.com'))                                   return 'rumble';
    if (u.includes('odysee.com'))                                   return 'odysee';
    if (u.includes('kick.com'))                                     return 'kick';
    if (u.includes('nebula.tv'))                                    return 'nebula';
    if (u.includes('archive.org'))                                  return 'archive';
    if (u.includes('/videos/embed/') || u.includes('/videos/watch/')) return 'peertube';
    return 'unknown';
  }

  /* ── Duration formatter ───────────────────────────────────────── */

  function _fmtDuration(seconds) {
    if (!seconds && seconds !== 0) return '';
    const s = Math.round(Number(seconds));
    if (isNaN(s) || s < 0) return '';
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
    return `${m}:${String(sec).padStart(2, '0')}`;
  }

  /* ── Card rendering ───────────────────────────────────────────── */

  function renderCard(video, opts) {
    opts = opts || {};
    const platform = detectPlatform(video.url || video.link || '');
    const p = PLATFORMS[platform] || PLATFORMS.unknown;

    // Thumbnail
    const thumb = video.thumbnail || video.thumbnailUrl || '';
    const thumbInner = thumb
      ? `<img src="${esc(thumb)}" alt="" loading="lazy" decoding="async">`
      : `<div class="mc-thumb-placeholder">▶</div>`;

    // Duration badge
    const dur = _fmtDuration(video.duration || video.length);
    const durBadge = dur ? `<span class="mc-duration">${esc(dur)}</span>` : '';

    // Platform badge
    const platBadge = `<span class="mc-badge" style="background:${p.color}">${p.icon} ${esc(p.label)}</span>`;

    // Meta line
    const parts = [];
    if (video.channel || video.author) parts.push(esc(video.channel || video.author));
    if (video.views != null)            parts.push(esc(String(video.views)) + ' views');
    if (video.published || video.date)  parts.push(esc(video.published || video.date));
    const meta = parts.join(' · ');

    // Cluster tag
    const cluster = video.cluster_name
      ? `<span class="mc-cluster">${esc(video.cluster_name)}</span>`
      : '';

    // Action buttons
    let actions = '';
    if (!opts.hideActions) {
      actions = `<div class="mc-actions">
        <button class="mc-action" data-action="play" title="Play">▶</button>
        <button class="mc-action" data-action="ambient" title="Send to Ambient">♫</button>
        <button class="mc-action" data-action="note" title="Save as Note">📝</button>
        <button class="mc-action" data-action="bookmark" title="Save to Files">📂</button>
        <button class="mc-action" data-action="queue" title="Add to Queue">⏭</button>
      </div>`;
    }

    // Encode video data for the attribute (double-escape for safe embedding)
    const dataJson = JSON.stringify(video).replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    return `<div class="media-card" data-video='${dataJson}' data-platform="${esc(platform)}">
      <div class="mc-thumb">
        ${thumbInner}
        ${durBadge}
        ${platBadge}
      </div>
      <div class="mc-body">
        <div class="mc-title">${esc(video.title || 'Untitled')}</div>
        <div class="mc-meta">${meta}</div>
        ${cluster}
      </div>
      ${actions}
    </div>`;
  }

  /* ── Action wiring (event delegation) ─────────────────────────── */

  function wireActions(container) {
    if (!container) return;
    container.addEventListener('click', function (e) {
      const actionBtn = e.target.closest('.mc-action[data-action]');
      const card = e.target.closest('.media-card');
      if (!card) return;

      let video;
      try { video = JSON.parse(card.getAttribute('data-video')); } catch (_) { return; }

      if (actionBtn) {
        e.stopPropagation();
        const action = actionBtn.dataset.action;
        if (action === 'play') {
          container.dispatchEvent(new CustomEvent('media:play', { detail: video, bubbles: true }));
        } else if (action === 'ambient') {
          container.dispatchEvent(new CustomEvent('media:send-to-ambient', { detail: video, bubbles: true }));
        } else if (action === 'note') {
          container.dispatchEvent(new CustomEvent('media:save-to-note', { detail: video, bubbles: true }));
        } else if (action === 'bookmark') {
          container.dispatchEvent(new CustomEvent('media:save-to-files', { detail: video, bubbles: true }));
        } else if (action === 'queue') {
          _addToQueue(video);
        }
        return;
      }

      // Card click (not on action) → play
      container.dispatchEvent(new CustomEvent('media:play', { detail: video, bubbles: true }));
    });
  }

  /* ── Queue management (localStorage) ──────────────────────────── */

  const QUEUE_KEY = 'mediaQueue';
  const QUEUE_MAX = 50;

  function _readQueue() {
    try { return JSON.parse(localStorage.getItem(QUEUE_KEY)) || []; }
    catch (_) { return []; }
  }

  function _saveQueue(queue) {
    localStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
    window.dispatchEvent(new CustomEvent('media:queue-updated'));
  }

  function _addToQueue(video) {
    const queue = _readQueue();
    const id = video.url || video.videoId || video.link;
    const exists = queue.some(v => (v.url || v.videoId || v.link) === id);
    if (exists) return;
    queue.push(video);
    if (queue.length > QUEUE_MAX) queue.splice(0, queue.length - QUEUE_MAX);
    _saveQueue(queue);
  }

  function getQueue() {
    return _readQueue();
  }

  function removeFromQueue(index) {
    const queue = _readQueue();
    if (index >= 0 && index < queue.length) {
      queue.splice(index, 1);
      _saveQueue(queue);
    }
  }

  function clearQueue() {
    _saveQueue([]);
  }

  /* ── Public API ───────────────────────────────────────────────── */

  window.MediaCards = {
    PLATFORMS,
    detectPlatform,
    renderCard,
    wireActions,
    getQueue,
    removeFromQueue,
    clearQueue,
  };
})();
