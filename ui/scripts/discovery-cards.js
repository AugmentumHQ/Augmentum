/**
 * Discovery Cards — Editorial bento renderer for For You recommendations.
 * Per zone: 1 hero (large, cover-image + gradient overlay, editorial title),
 * 2 medium (thumb-left), the rest as peek chips (compact single-line).
 * Loaded as a regular script; exposes window.DiscoveryCards.
 */
(function () {
  'use strict';

  // Prefer the project-wide escapeHtml (defined in app.js) — it also
  // handles backticks and ${, which the DOM-textContent trick does not.
  const _esc = window.escapeHtml || ((s) => {
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  });

  function _extractYouTubeId(url) {
    if (!url) return null;
    try {
      const u = new URL(url);
      if (u.hostname.includes('youtube.com')) return u.searchParams.get('v') || null;
      if (u.hostname === 'youtu.be') return u.pathname.slice(1) || null;
    } catch { /* ignore */ }
    return null;
  }

  function _domainOf(item) {
    try { return item.domain || new URL(item.url || '').hostname; } catch { return ''; }
  }

  function _isVideo(item) {
    return item.content_type === 'video' || !!_extractYouTubeId(item.url);
  }

  function _thumbnailUrl(item) {
    let t = item.thumbnail || '';
    if (!t && _isVideo(item)) {
      const vid = _extractYouTubeId(item.url);
      if (vid) t = `https://img.youtube.com/vi/${vid}/hqdefault.jpg`;
    }
    return t ? `/api/browse/image?url=${encodeURIComponent(t)}` : '';
  }

  function _faviconHtml(domain) {
    if (!domain) return '';
    const url = `/api/browse/image?url=${encodeURIComponent(
      `https://www.google.com/s2/favicons?domain=${domain}&sz=32`
    )}`;
    return `<img class="discovery-card-favicon" src="${url}" alt="" width="14" height="14" loading="lazy" onerror="this.style.display='none'">`;
  }

  function _actionsHtml(item) {
    const url = _esc(item.url);
    const cluster = _esc(item.cluster_id || '');
    return `
      <button class="discovery-card-hide" title="Hide this page" data-url="${url}" aria-label="Hide">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
      </button>
      <button class="discovery-card-dismiss" title="Not interested in topic" data-cluster="${cluster}" aria-label="Dismiss topic">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    `;
  }

  function _sourceLine(item, ctx) {
    const cluster = item.cluster_name
      ? `<span class="discovery-card-cluster">${_esc(item.cluster_name)}</span>`
      : '';
    const ribbon = ctx.paper ? '<span class="discovery-card-ribbon">paper</span>' : '';
    return `
      <div class="discovery-card-source">
        ${_faviconHtml(ctx.domain)}
        <span>${_esc(ctx.domain)}</span>
        ${ribbon}
        ${cluster}
      </div>
    `;
  }

  // Compute the per-item values each renderer needs. Cheaper than reparsing
  // item.url 2-3 times across _sourceLine / _faviconHtml / _isPaper.
  function _itemCtx(item) {
    const domain = _domainOf(item);
    const paper = item.content_type === 'paper' || (domain || '').toLowerCase().includes('arxiv.org');
    return { domain, paper };
  }

  // ── Hero ──────────────────────────────────────────────────────────────

  function _renderHero(item) {
    const ctx = _itemCtx(item);
    const thumb = _thumbnailUrl(item);
    const video = _isVideo(item);
    const coverHtml = thumb
      ? `<div class="discovery-hero-cover">
           <img src="${thumb}" alt="" loading="lazy">
           ${video ? `<div class="discovery-hero-play">
               <svg viewBox="0 0 16 16" width="14" height="14" fill="#fff"><polygon points="4,2 14,8 4,14"/></svg>
             </div>` : ''}
         </div>`
      : `<div class="discovery-hero-cover discovery-hero-cover-empty"></div>`;
    const titleCls = ctx.paper ? 'discovery-hero-title discovery-hero-title-paper' : 'discovery-hero-title';
    const snippet = item.snippet
      ? `<div class="discovery-hero-snippet">${_esc(item.snippet)}</div>`
      : '';
    return `
      <div class="discovery-card discovery-card-hero" data-url="${_esc(item.url)}" data-cluster="${_esc(item.cluster_id || '')}">
        ${coverHtml}
        <div class="discovery-hero-scrim"></div>
        <div class="discovery-hero-body">
          ${_sourceLine(item, ctx)}
          <div class="${titleCls}">${_esc(item.title)}</div>
          ${snippet}
        </div>
        <div class="discovery-card-actions">${_actionsHtml(item)}</div>
      </div>
    `;
  }

  // ── Medium ────────────────────────────────────────────────────────────

  function _renderMedium(item) {
    const ctx = _itemCtx(item);
    const thumb = _thumbnailUrl(item);
    const video = _isVideo(item);
    const thumbHtml = thumb
      ? `<div class="discovery-card-thumb">
           <img src="${thumb}" alt="" loading="lazy">
           ${video ? '<div class="discovery-thumb-play"><svg viewBox="0 0 12 12" width="10" height="10" fill="#fff"><polygon points="3,1 10,6 3,11"/></svg></div>' : ''}
         </div>`
      : '';
    return `
      <div class="discovery-card discovery-card-medium" data-url="${_esc(item.url)}" data-cluster="${_esc(item.cluster_id || '')}">
        <div class="discovery-card-body">
          ${_sourceLine(item, ctx)}
          <div class="discovery-card-title">${_esc(item.title)}</div>
          ${item.snippet ? `<div class="discovery-card-snippet">${_esc(item.snippet)}</div>` : ''}
        </div>
        ${thumbHtml}
        <div class="discovery-card-actions">${_actionsHtml(item)}</div>
      </div>
    `;
  }

  // ── Peek (compact single-line) ────────────────────────────────────────

  function _renderPeek(item) {
    const ctx = _itemCtx(item);
    return `
      <div class="discovery-card discovery-card-peek" data-url="${_esc(item.url)}" data-cluster="${_esc(item.cluster_id || '')}">
        ${_faviconHtml(ctx.domain)}
        <div class="discovery-peek-title">${_esc(item.title)}</div>
        ${ctx.paper ? '<span class="discovery-card-ribbon">paper</span>' : ''}
        <span class="discovery-peek-domain">${_esc(ctx.domain)}</span>
        <div class="discovery-card-actions">${_actionsHtml(item)}</div>
      </div>
    `;
  }

  // ── Back-compat entry point ───────────────────────────────────────────

  function renderRecommendationCard(item, size) {
    if (size === 'hero')   return _renderHero(item);
    if (size === 'peek')   return _renderPeek(item);
    return _renderMedium(item);
  }

  // ── Zone sectioning ───────────────────────────────────────────────────

  // Editorial sentence-case headers — less "system panel," more "curator."
  const _ZONE_HEADERS = {
    core:     { kicker: 'In your orbit',        subtitle: "picking up where you left off" },
    frontier: { kicker: 'A step beyond',        subtitle: "near what you know, not quite inside it" },
    adjacent: { kicker: 'Off the path',         subtitle: "things outside your usual patterns" },
    fresh:    { kicker: 'Fresh from the feeds', subtitle: "just landed" },
  };

  function renderZoneSection(zoneName, items) {
    if (!items || items.length === 0) return '';
    const header = _ZONE_HEADERS[zoneName] || { kicker: zoneName, subtitle: '' };

    // Composition: first item = hero, next two = medium, rest = peek.
    // Zones with only 1-2 items fall through cleanly (hero + optional medium).
    const parts = [];
    items.forEach((item, idx) => {
      if (idx === 0) parts.push(_renderHero(item));
      else if (idx < 3) parts.push(_renderMedium(item));
      else parts.push(_renderPeek(item));
    });

    return `
      <section class="discovery-zone" data-zone="${_esc(zoneName)}">
        <header class="discovery-zone-header">
          <span class="discovery-zone-kicker">${_esc(header.kicker)}</span>
          ${header.subtitle ? `<span class="discovery-zone-subtitle">${_esc(header.subtitle)}</span>` : ''}
        </header>
        <div class="discovery-zone-bento">${parts.join('')}</div>
      </section>
    `;
  }

  window.DiscoveryCards = { renderRecommendationCard, renderZoneSection };
})();
