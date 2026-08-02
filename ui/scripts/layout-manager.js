/**
 * LayoutManager — arranges surface containers on screen.
 *
 * Phone OS model: one surface fills the screen at a time.
 * A tab bar shows all open surfaces for switching/closing.
 * Tab bar is hidden when only one surface is open.
 */

import { SurfaceRegistry } from './surface-registry.js';

let _grid = null;
let _tabBar = null;
const _containers = new Map();  // surfaceId → container element
const _tabs = new Map();        // surfaceId → tab element

export const LayoutManager = {
  init() {
    _grid = document.getElementById('surface-grid');
    if (!_grid) {
      console.error('LayoutManager: #surface-grid not found');
      return;
    }

    // Create the tab bar (inserted before the grid)
    _tabBar = document.createElement('div');
    _tabBar.className = 'surface-tab-bar hidden';
    _grid.parentElement.insertBefore(_tabBar, _grid);

    // Focus tracking
    document.addEventListener('surface:focus-changed', (e) => {
      const focusedId = e.detail.surfaceId;
      // Bail if the focused id has no mounted container — otherwise every
      // container would be set data-focused="false" and the screen would go
      // blank. Can happen during workspace restore/teardown races.
      if (focusedId && !_containers.has(focusedId)) return;
      // Show only the focused container
      _containers.forEach((container, id) => {
        container.dataset.focused = (id === focusedId) ? 'true' : 'false';
      });
      // Update tab active state AND scroll the now-active tab into view.
      // On a narrow viewport (360-390px) the tab bar is `overflow-x: auto`
      // with up to 4 tabs of max-width 160px, meaning 2-3 tabs can live
      // off-screen. Without scrolling-into-view a user who focuses one
      // of those hidden tabs would see the correct surface but a tab
      // bar that looks unchanged — classic "is it broken?" UX. Use
      // `block: 'nearest'` so we don't disturb vertical scroll and
      // `inline: 'nearest'` to avoid re-centering when the active tab
      // is already visible.
      _tabs.forEach((tab, id) => {
        const isActive = id === focusedId;
        tab.classList.toggle('active', isActive);
        if (isActive && typeof tab.scrollIntoView === 'function') {
          try {
            tab.scrollIntoView({
              behavior: 'smooth',
              inline: 'nearest',
              block: 'nearest',
            });
          } catch { /* older engines without options object */ }
        }
      });
    });
  },

  mountSurface(surface, options = {}) {
    if (!_grid) return;

    // --- Container ---
    const container = document.createElement('div');
    container.className = 'surface-container';
    container.dataset.surfaceId = surface.id;
    container.dataset.surfaceType = surface.constructor.type;
    container.dataset.focused = 'false';

    const orbColor = surface.getIcon();
    container.style.setProperty('--surface-orb-color', `var(--orb-${orbColor})`);
    container.style.setProperty('--surface-focus-color', `var(--orb-${orbColor})`);

    // Content area (surface mounts into this)
    const content = document.createElement('div');
    content.className = 'surface-content';
    container.appendChild(content);
    _grid.appendChild(container);
    _containers.set(surface.id, container);

    // Mount the surface
    surface.mount(content);

    // --- Tab ---
    const tab = document.createElement('button');
    tab.className = 'surface-tab';
    tab.dataset.surfaceId = surface.id;
    tab.style.setProperty('--tab-color', `var(--orb-${orbColor})`);

    // Only render the close button when the surface is actually safe to
    // close. Surfaces that adopt singleton DOM (primary chat/narrative,
    // all coder) glitch main-area's flex layout when unmounted while
    // another tab is open — hiding the button removes the footgun.
    const closeable = typeof surface.isCloseable === 'function' ? surface.isCloseable() : true;
    const closeMarkup = closeable
      ? '<span class="surface-tab-close" title="Close">&times;</span>'
      : '';
    tab.innerHTML =
      '<span class="surface-tab-dot"></span>' +
      '<span class="surface-tab-title">' + surface.getTitle() + '</span>' +
      closeMarkup;

    // Click tab → focus surface
    tab.addEventListener('click', (e) => {
      if (e.target.closest('.surface-tab-close')) return;
      SurfaceRegistry.focus(surface.id);
    });

    // Close button (wired only when present)
    const closeBtn = tab.querySelector('.surface-tab-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.closeSurface(surface.id);
      });
    }

    _tabs.set(surface.id, tab);
    _tabBar.appendChild(tab);

    // Listen for title changes
    surface.on('surface:titleChanged', () => {
      const titleEl = tab.querySelector('.surface-tab-title');
      if (titleEl) titleEl.textContent = surface.getTitle();
    });

    // Focus and update
    SurfaceRegistry.focus(surface.id);
    this._updateTabBar();
  },

  /**
   * Canonical user-facing close: unmount + destroy + persist. Every "close
   * this tab" path (tab close button, command-composer surface.close, voice
   * "close this") must come through here. Persisting matters: without the
   * saveWorkspace a closed surface resurrects on the next page load — a real
   * bug with coder tabs that used to reappear every reload. In-place swap
   * paths (ViewStack primary swap) call unmountSurface directly instead,
   * because saving mid-swap would persist a half-torn-down layout.
   */
  closeSurface(surfaceId) {
    this.unmountSurface(surfaceId);
    SurfaceRegistry.destroy(surfaceId);
    this._updateTabBar();
    SurfaceRegistry.saveWorkspace();
  },

  unmountSurface(surfaceId, options = {}) {
    const container = _containers.get(surfaceId);
    if (!container) return;

    const surface = SurfaceRegistry.get(surfaceId);
    if (surface) surface.unmount();

    // Remove container
    container.remove();
    _containers.delete(surfaceId);

    // Remove tab
    const tab = _tabs.get(surfaceId);
    if (tab) tab.remove();
    _tabs.delete(surfaceId);

    // Focus next surface unless the caller is replacing this surface in-place.
    // Primary mode swaps mount the destination immediately; focusing a random
    // survivor in the middle can make singleton chat DOM reconcile against the
    // wrong surface for one frame.
    if (!options.preserveFocus) {
      const focused = SurfaceRegistry.getFocused();
      if (!focused || focused.id === surfaceId) {
        const remaining = [..._containers.keys()];
        if (remaining.length > 0) {
          SurfaceRegistry.focus(remaining[remaining.length - 1]);
        }
      }
    }

    this._updateTabBar();
  },

  _updateTabBar() {
    if (!_tabBar) return;
    // Show tab bar only when multiple surfaces are open
    const count = _containers.size;
    _tabBar.classList.toggle('hidden', count <= 1);
  },

  get count() { return _containers.size; },
  get isMobile() { return window.innerWidth < 768; },
  hasContainer(surfaceId) { return _containers.has(surfaceId); },
};
