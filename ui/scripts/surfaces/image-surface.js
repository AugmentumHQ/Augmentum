import { Surface } from '../surface.js';
import { SurfaceRegistry } from '../surface-registry.js';

/**
 * ImageSurface — image generation + gallery, wrapped as a surface.
 *
 * Status: REGISTERED BUT NOT YET REACHABLE. No orb maps to 'image', and no
 * callsite runs SurfaceRegistry.create('image'). The skeleton is here because
 * the surface architecture spec treats image as a first-class singleton
 * surface (see docs/superpowers/specs/2026-04-08-surface-architecture-design.md
 * §ImageSurface and §Surface Templates), and when the image feature gets its
 * own orb or a "Creative" template ships, this will be the mount point.
 *
 * Treated as single-instance by orb-nav._SINGLE_INSTANCE_SURFACES.
 */
export class ImageSurface extends Surface {
  static type = 'image';

  constructor(id, config = {}) {
    super(id, config);
  }

  mount(container) {
    super.mount(container);
    const imagePanel = document.getElementById('image-panel');
    if (imagePanel) {
      imagePanel.classList.remove('hidden');
      // Remove fixed positioning so it flows in the grid
      imagePanel.style.position = 'relative';
      imagePanel.style.width = '100%';
      imagePanel.style.height = '100%';
      container.appendChild(imagePanel);
    }
  }

  unmount() {
    if (this._container) {
      const imagePanel = this._container.querySelector('#image-panel');
      if (imagePanel) {
        const appEl = document.getElementById('app');
        if (appEl) appEl.appendChild(imagePanel);
        imagePanel.classList.add('hidden');
        // Restore fixed positioning
        imagePanel.style.position = '';
        imagePanel.style.width = '';
        imagePanel.style.height = '';
      }
    }
    super.unmount();
  }

  getTitle() { return 'Image'; }
  getIcon() { return 'agentic'; } // orange orb for image/creative

  getContext() {
    return {
      type: 'image',
      id: this.id,
      summary: 'Image generation',
      capabilities: ['generate', 'img2img', 'inpaint', 'upscale', 'remove-bg'],
    };
  }
}

SurfaceRegistry.register('image', ImageSurface);
