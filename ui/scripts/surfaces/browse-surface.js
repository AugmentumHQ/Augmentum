import { Surface } from '../surface.js';
import { SurfaceRegistry } from '../surface-registry.js';

export class BrowseSurface extends Surface {
  static type = 'browse';

  constructor(id, config = {}) {
    super(id, config);
    this.url = config.url || '';
    this.pageTitle = config.pageTitle || '';
  }

  mount(container) {
    super.mount(container);
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    const browsePanel = document.getElementById('browse-panel');
    if (browsePanel) {
      browsePanel.classList.remove('hidden');
      // Override fixed positioning — flow in surface container
      browsePanel.style.position = 'relative';
      browsePanel.style.top = 'auto';
      browsePanel.style.left = 'auto';
      browsePanel.style.right = 'auto';
      browsePanel.style.bottom = 'auto';
      browsePanel.style.width = '100%';
      browsePanel.style.height = '100%';
      browsePanel.style.zIndex = 'auto';
      container.appendChild(browsePanel);
    }

    // Workspace restore: if we have a pinned URL from a previous session,
    // re-navigate to it. Uses the existing chat→browse bridge event so
    // all the usual side effects (history stack, reader view, state
    // bookkeeping inside browse.js) happen correctly — much safer than
    // duplicating that logic here.
    if (this.url) {
      document.dispatchEvent(new CustomEvent('augmentum:browse-url', {
        detail: { url: this.url },
      }));
    }
  }

  unmount() {
    if (this._container) {
      const browsePanel = this._container.querySelector('#browse-panel');
      if (browsePanel) {
        const appEl = document.getElementById('app');
        if (appEl) appEl.appendChild(browsePanel);
        browsePanel.classList.add('hidden');
        // Restore original positioning
        browsePanel.style.position = '';
        browsePanel.style.top = '';
        browsePanel.style.left = '';
        browsePanel.style.right = '';
        browsePanel.style.bottom = '';
        browsePanel.style.width = '';
        browsePanel.style.height = '';
        browsePanel.style.zIndex = '';
      }
    }
    super.unmount();
  }

  getTitle() { return this.pageTitle || 'Browse'; }
  getIcon() { return 'browse'; }

  getContext() {
    return {
      type: 'browse',
      id: this.id,
      summary: this.pageTitle ? `Reading: ${this.pageTitle}` : 'Browser',
      url: this.url,
      capabilities: ['read', 'extract', 'search', 'note'],
    };
  }

  getState() {
    return {
      ...super.getState(),
      url: this.url,
      pageTitle: this.pageTitle,
    };
  }

  restoreState(state) {
    super.restoreState(state);
    if (state.url) this.url = state.url;
    if (state.pageTitle) this.pageTitle = state.pageTitle;
  }
}

SurfaceRegistry.register('browse', BrowseSurface);
