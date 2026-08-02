import { Surface } from '../surface.js';
import { SurfaceRegistry } from '../surface-registry.js';
import {
  startCoderPermissionListener,
  stopCoderPermissionListener,
} from '../coder-permissions.js';

export class CoderSurface extends Surface {
  static type = 'coder';

  constructor(id, config = {}) {
    super(id, config);
    // workspaceId/workspaceName are architectural placeholders — CoderSurface
    // is currently single-instance and coder.js owns the active workspace in
    // its own module-level `_activeWorkspaceId`. When multi-workspace coder
    // tabs become a real feature (see surface-architecture-design.md
    // §CoderSurface), these fields will be populated on construction and the
    // tab title will reflect the per-surface workspace. For now they
    // round-trip through getState/restoreState but only affect getTitle.
    this.workspaceId = config.workspaceId || '';
    this.workspaceName = config.workspaceName || 'Workspace';
    // Matches ChatSurface/NarrativeSurface: primary surfaces represent the
    // active mode and are swapped out on mode change, not closed via X. Only
    // drag-and-drop-created alongside coders (non-primary) expose a close
    // button.
    this._isPrimary = config.primary || false;
  }

  mount(container) {
    super.mount(container);
    container.style.display = 'flex';
    container.style.flexDirection = 'column';

    // DOM adoption note: we intentionally adopt NOTHING here.
    //
    // History: the surface was designed to adopt coder's singleton DOM
    // cluster (terminal pane, editor split, status bar, intent bar) into
    // .surface-content, mirroring how ChatSurface adopts #chat-scroll.
    // That pattern breaks for coder mode because #surface-grid is
    // display:none in coder mode (coder.css:55) — adopted elements end
    // up in a collapsed ancestor with 0x0 bounds, invisible to the user
    // even though classes and innerHTML are correct. Editor-split was
    // removed from the adoption list on 2026-04-22 for exactly this
    // reason; now we remove the other three for a second reason:
    //
    // Race fix (refresh path, 2026-04-22): Coder.init() is fire-and-
    // forget from app.js bootstrap (line ~4151), and its end-of-init
    // double-RAF can resolve BEFORE `await _bootSurfaces()` finishes
    // mounting this surface (~4385). If mount() adopts DOM here, the
    // order becomes:
    //   1. _onEnterCoderMode reparents pane to coder-terminal-wrapper,
    //      Terminal.create runs, xterm attaches cleanly.
    //   2. Surface mount completes LATER, steals the pane (with its
    //      attached xterm) into .surface-content (display:none).
    // Result: terminal exists but renders at 0x0, user sees an empty
    // pane. Observed 2026-04-22 via devtools: `pane parent` =
    // "surface-content", `pane has xterm` = true, `pane rect` = 0x0.
    // By NOT adopting, we eliminate the race — the pane never enters
    // surface-content and _onEnterCoderMode owns its lifecycle alone.
    //
    // The surface remains a logical container for tab tracking, focus
    // state, and permission listener lifecycle; it just doesn't own
    // DOM. If chat-style flex:1 "glitched splitscreen" re-emerges on
    // a mode switch (see the old unmount comment), the fix is at the
    // applyMode() visibility toggle, not here.

    // Start listening for tool-permission approval requests whenever the
    // coder surface is mounted. The listener is a no-op when the backend
    // policy is AUGMENTUM_CODER_PERMISSIONS=auto (no pending requests
    // will ever be returned).
    startCoderPermissionListener();
  }

  unmount() {
    stopCoderPermissionListener();

    // Symmetric with mount() — we don't own any coder DOM, so there's
    // nothing to return to main-area. The pane, status bar, intent
    // bar, and editor split all live under .main-area / #coder-layout
    // and are hidden by applyMode() via the `hidden` class when the
    // mode switches away. See the DOM-adoption note in mount() for
    // the race-fix rationale.

    super.unmount();
  }

  getTitle() { return this.workspaceName || 'Code'; }
  getIcon() { return 'coder'; }

  // Non-primary coder tabs (drag-dropped alongside) expose a close button.
  // The primary coder surface represents the active mode; leaving coder
  // mode swaps the primary (see ViewStack._swapPrimaryForMode), so an X on
  // the primary would orphan the UI with no surface focused. Unmount re-
  // applies `hidden` to returned DOM in both cases.
  isCloseable() { return !this._isPrimary; }

  getContext() {
    return {
      type: 'coder',
      id: this.id,
      mode: 'coder',
      summary: `Workspace: ${this.workspaceName}`,
      capabilities: ['code-edit', 'terminal', 'git', 'file-ops'],
    };
  }

  getState() {
    return {
      ...super.getState(),
      workspaceId: this.workspaceId,
      workspaceName: this.workspaceName,
      primary: this._isPrimary,
    };
  }

  restoreState(state) {
    super.restoreState(state);
    if (state.workspaceId) this.workspaceId = state.workspaceId;
    if (state.workspaceName) this.workspaceName = state.workspaceName;
    if (state.primary !== undefined) this._isPrimary = state.primary;
  }
}

SurfaceRegistry.register('coder', CoderSurface);
