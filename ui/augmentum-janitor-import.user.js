// ==UserScript==
// @name         Augmentum — JanitorAI Importer
// @namespace    augmentum
// @version      2.0
// @description  Adds a "Send to Augmentum" button on JanitorAI character pages
// @author       Augmentum
// @match        https://janitorai.com/characters/*
// @match        https://www.janitorai.com/characters/*
// @grant        none
// @run-at       document-start
// ==/UserScript==

(function () {
  'use strict';

  // --- Storage ---
  const STORAGE_KEY = 'augmentum_url';
  const MODE_KEY = 'augmentum_import_mode'; // 'send' or 'copy'
  const getAugUrl = () => localStorage.getItem(STORAGE_KEY) || '';
  const setAugUrl = (url) => localStorage.setItem(STORAGE_KEY, url);
  const getMode = () => localStorage.getItem(MODE_KEY) || 'send';
  const setMode = (mode) => localStorage.setItem(MODE_KEY, mode);

  // --- Character data capture ---
  // We intercept the page's own network requests to capture character data
  // that JanitorAI fetches from its API (avoids CORS and Cloudflare entirely)
  let capturedCharacter = null;

  function isCharacterData(obj) {
    return obj && typeof obj === 'object' &&
      typeof obj.name === 'string' && obj.name.length > 0 && obj.name.length < 200 &&
      (obj.description || obj.personality || obj.first_message || obj.scenario);
  }

  // Recursively search an object for character-like data
  function findCharacterData(obj, depth) {
    if (!obj || typeof obj !== 'object' || depth > 6) return null;
    if (isCharacterData(obj)) return obj;
    if (Array.isArray(obj)) {
      for (let i = 0; i < Math.min(obj.length, 20); i++) {
        const r = findCharacterData(obj[i], depth + 1);
        if (r) return r;
      }
    } else {
      for (const k of Object.keys(obj)) {
        const r = findCharacterData(obj[k], depth + 1);
        if (r) return r;
      }
    }
    return null;
  }

  // Track current page URL to detect SPA navigation
  let lastPathname = location.pathname;

  function checkNavigation() {
    if (location.pathname !== lastPathname) {
      lastPathname = location.pathname;
      // New page — clear captured data so fresh data takes over
      if (location.pathname.includes('/characters/')) {
        capturedCharacter = null;
        console.log('[Augmentum] Page navigation detected, waiting for new character data...');
        if (btnEl) {
          btnEl.textContent = '\u23F3 Waiting for character data...';
          btnEl.style.opacity = '0.6';
          btnEl.disabled = true;
        }
      }
    }
  }

  // Poll for URL changes (SPA pushState doesn't fire events reliably)
  setInterval(checkNavigation, 500);

  // Also hook pushState/replaceState for immediate detection
  const origPushState = history.pushState;
  const origReplaceState = history.replaceState;
  history.pushState = function (...args) {
    origPushState.apply(this, args);
    checkNavigation();
  };
  history.replaceState = function (...args) {
    origReplaceState.apply(this, args);
    checkNavigation();
  };
  window.addEventListener('popstate', checkNavigation);

  // Strategy 1: Intercept window.fetch responses
  const originalFetch = window.fetch;
  window.fetch = async function (...args) {
    const response = await originalFetch.apply(this, args);
    try {
      const url = typeof args[0] === 'string' ? args[0] : args[0]?.url || '';
      if (url.includes('/characters/') || url.includes('character')) {
        const clone = response.clone();
        clone.json().then(data => {
          const charData = findCharacterData(data, 0);
          if (charData) {
            // Always update — allows switching characters without refresh
            const isNew = !capturedCharacter || capturedCharacter.name !== charData.name;
            capturedCharacter = charData;
            capturedCharacter._source = 'fetch_intercept';
            if (isNew) console.log('[Augmentum] Captured character data from fetch:', charData.name);
            updateButtonState();
          }
        }).catch(() => {});
      }
    } catch (_) {}
    return response;
  };

  // Strategy 2: Intercept XMLHttpRequest responses
  const originalXHROpen = XMLHttpRequest.prototype.open;
  const originalXHRSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this._augUrl = url;
    return originalXHROpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (...args) {
    this.addEventListener('load', function () {
      try {
        if (this._augUrl && (this._augUrl.includes('/characters/') || this._augUrl.includes('character'))) {
          const data = JSON.parse(this.responseText);
          const charData = findCharacterData(data, 0);
          if (charData) {
            const isNew = !capturedCharacter || capturedCharacter.name !== charData.name;
            capturedCharacter = charData;
            capturedCharacter._source = 'xhr_intercept';
            if (isNew) console.log('[Augmentum] Captured character data from XHR:', charData.name);
            updateButtonState();
          }
        }
      } catch (_) {}
    });
    return originalXHRSend.apply(this, args);
  };

  // Strategy 3: Check __NEXT_DATA__ after page loads
  function tryNextData() {
    const el = document.getElementById('__NEXT_DATA__');
    if (!el) return null;
    try {
      const nd = JSON.parse(el.textContent);
      return findCharacterData(nd, 0);
    } catch (_) {
      return null;
    }
  }

  // Strategy 4: Try to find data in React Query cache or window globals
  function tryReactQueryCache() {
    // React Query stores data in a QueryClient, often attached to window or React context
    // Check common patterns
    try {
      // Some apps expose the cache globally
      if (window.__REACT_QUERY_STATE__) {
        return findCharacterData(window.__REACT_QUERY_STATE__, 0);
      }
      // Check for Next.js data in various locations
      if (window.__NEXT_DATA__) {
        return findCharacterData(window.__NEXT_DATA__, 0);
      }
    } catch (_) {}
    return null;
  }

  // --- Send to Augmentum ---
  function sendToAugmentum(data) {
    let augUrl = getAugUrl();
    if (!augUrl) {
      augUrl = prompt(
        'Enter your Augmentum URL:\n(e.g., https://localhost:6443 or https://myserver.duckdns.org:6443)\n\nThis is saved for future use.',
        'https://localhost:6443'
      );
      if (!augUrl) return;
      setAugUrl(augUrl);
    }

    // Add UUID from current page
    const uuidMatch = location.pathname.match(
      /\/characters\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i
    );
    if (uuidMatch) data._janitor_uuid = uuidMatch[1];

    const encoded = encodeURIComponent(JSON.stringify(data));
    const url = augUrl.replace(/\/+$/, '') + '/ui/#janitor-import=' + encoded;

    if (url.length > 60000) {
      // Large payload — use postMessage
      const w = window.open(augUrl.replace(/\/+$/, '') + '/ui/#janitor-pending', '_blank');
      if (!w) { alert('Pop-up blocked. Please allow pop-ups for this site.'); return; }
      let done = false, attempts = 0;
      const iv = setInterval(() => {
        if (done) { clearInterval(iv); return; }
        if (++attempts > 120) { clearInterval(iv); alert('Timed out sending to Augmentum. Make sure the page finished loading and try again.'); return; }
        try { w.postMessage({ type: 'janitor-import', data }, '*'); } catch (_) {}
      }, 500);
      window.addEventListener('message', e => {
        if (e.data && e.data.type === 'janitor-ack') done = true;
      });
    } else {
      window.open(url, '_blank');
    }
  }

  // --- Copy JSON to clipboard ---
  function copyJsonToClipboard(data) {
    const json = JSON.stringify(data, null, 2);
    navigator.clipboard.writeText(json).then(() => {
      btnEl.textContent = '\u2713 Copied to clipboard!';
      setTimeout(() => updateButtonState(), 2000);
    }).catch(() => {
      // Fallback: prompt with text
      prompt('Copy this JSON:', json);
    });
  }

  // --- UI ---
  let btnEl = null;
  let toggleEl = null;

  function updateButtonState() {
    if (!btnEl) return;
    const mode = getMode();
    if (capturedCharacter) {
      const shortName = capturedCharacter.name.length > 20
        ? capturedCharacter.name.slice(0, 20) + '\u2026'
        : capturedCharacter.name;
      btnEl.textContent = mode === 'copy'
        ? '\uD83D\uDCCB Copy "' + shortName + '" JSON'
        : '\u2B06 Send "' + shortName + '" to Augmentum';
      btnEl.style.opacity = '1';
      btnEl.disabled = false;
    }
    if (toggleEl) {
      toggleEl.textContent = mode === 'copy' ? '\uD83D\uDCCB Copy' : '\u2B06 Send';
      toggleEl.title = mode === 'copy'
        ? 'Mode: Copy JSON to clipboard. Click to switch to Send.'
        : 'Mode: Send directly to Augmentum. Click to switch to Copy.';
    }
  }

  function createButton() {
    // Container
    const wrap = document.createElement('div');
    wrap.style.cssText = `
      position: fixed; bottom: 20px; right: 20px; z-index: 99999;
      display: flex; gap: 0; border-radius: 8px; overflow: hidden;
      box-shadow: 0 4px 16px rgba(0,0,0,0.3);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    `;

    // Main button
    const btn = document.createElement('button');
    btn.id = 'augmentum-import-btn';
    btn.textContent = '\u23F3 Waiting for character data...';
    btn.style.cssText = `
      padding: 10px 18px; border: none;
      background: linear-gradient(135deg, #7b68ee, #4fc3f7);
      color: white; font-weight: 600; font-size: 14px;
      cursor: pointer;
      transition: filter 0.15s, opacity 0.3s;
      opacity: 0.6;
    `;
    btn.disabled = true;
    btn.addEventListener('mouseenter', () => { btn.style.filter = 'brightness(1.1)'; });
    btn.addEventListener('mouseleave', () => { btn.style.filter = ''; });
    btn.addEventListener('click', () => {
      if (!capturedCharacter) {
        alert('Character data has not been captured yet.\n\n' +
          'This can happen if the page loaded before the userscript.\n' +
          'Try refreshing the page (F5).');
        return;
      }
      const data = { ...capturedCharacter };
      if (getMode() === 'copy') {
        copyJsonToClipboard(data);
      } else {
        sendToAugmentum(data);
        btn.textContent = '\u2713 Sent!';
        setTimeout(() => updateButtonState(), 2000);
      }
    });

    // Mode toggle pill
    const toggle = document.createElement('button');
    toggle.style.cssText = `
      padding: 10px 10px; border: none; border-left: 1px solid rgba(255,255,255,0.2);
      background: linear-gradient(135deg, #6a5acd, #3db8e0);
      color: white; font-weight: 600; font-size: 11px;
      cursor: pointer;
      transition: filter 0.15s;
      white-space: nowrap;
    `;
    toggle.addEventListener('mouseenter', () => { toggle.style.filter = 'brightness(1.15)'; });
    toggle.addEventListener('mouseleave', () => { toggle.style.filter = ''; });
    toggle.addEventListener('click', (e) => {
      e.stopPropagation();
      setMode(getMode() === 'send' ? 'copy' : 'send');
      updateButtonState();
    });
    toggleEl = toggle;

    wrap.appendChild(btn);
    wrap.appendChild(toggle);
    return { wrap, btn };
  }

  // --- Init ---
  // Wait for DOM to be ready before inserting the button
  function init() {
    const { wrap, btn } = createButton();
    btnEl = btn;
    document.body.appendChild(wrap);

    // Try static sources (page already loaded before our hook)
    if (!capturedCharacter) {
      const nd = tryNextData();
      if (nd) {
        capturedCharacter = nd;
        capturedCharacter._source = '__NEXT_DATA__';
        console.log('[Augmentum] Found character in __NEXT_DATA__:', nd.name);
      }
    }
    if (!capturedCharacter) {
      const rq = tryReactQueryCache();
      if (rq) {
        capturedCharacter = rq;
        capturedCharacter._source = 'react_query_cache';
        console.log('[Augmentum] Found character in React Query cache:', rq.name);
      }
    }
    updateButtonState();

    // If still no data, poll for it (page might load data after initial render)
    if (!capturedCharacter) {
      let pollCount = 0;
      const pollIv = setInterval(() => {
        pollCount++;
        if (capturedCharacter || pollCount > 30) { clearInterval(pollIv); return; }
        const nd = tryNextData();
        if (nd) { capturedCharacter = nd; capturedCharacter._source = '__NEXT_DATA__'; }
        const rq = tryReactQueryCache();
        if (rq) { capturedCharacter = rq; capturedCharacter._source = 'react_query_cache'; }
        if (capturedCharacter) {
          console.log('[Augmentum] Found character data (poll):', capturedCharacter.name);
          updateButtonState();
          clearInterval(pollIv);
        }
      }, 1000);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
