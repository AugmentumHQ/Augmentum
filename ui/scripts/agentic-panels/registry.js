/* ==========================================================================
   Agentic per-flow renderer registry

   Maps a flow id → a renderer object that owns the ``#task-flow-body`` slot
   for the duration of a flow's execution. The generic agentic panel still
   renders the standard pipeline / artifact / approval surfaces; the
   per-flow renderer lives BELOW those, providing flow-tailored UI
   (chapter strip for storybook, file tree for app builder, etc.).

   Renderer contract:

     {
       id: 'flow_research_illustrate',
       reset(slot) { ... }        // called when a new task with this flow starts
       handle(slot, meta) { ... } // called for every meta event during the run
     }

   - ``slot`` is the ``#task-flow-body`` DOM element. The renderer owns its
     contents; the registry never touches them.
   - ``handle`` should be cheap and idempotent — it fires on every chunk,
     including bare content deltas, so it must short-circuit when nothing
     interesting changed.
   ========================================================================== */

const _renderers = new Map();
let _activeRenderer = null;
let _activeTaskId = null;

export function register(renderer) {
  if (!renderer || !renderer.id) return;
  _renderers.set(renderer.id, renderer);
}

/** Pick a renderer for the meta envelope and hand it the slot.
 *  Returns true if a per-flow renderer handled this meta, false otherwise.
 */
export function dispatch(meta) {
  const slot = document.getElementById('task-flow-body');
  if (!slot) return false;
  const flowId = meta && meta.flow_id;
  const taskId = meta && meta.task_id;

  // New task: tear down whatever was in the slot, pick the new renderer.
  if (taskId && taskId !== _activeTaskId) {
    _activeTaskId = taskId;
    _activeRenderer = flowId ? _renderers.get(flowId) || null : null;
    slot.innerHTML = '';
    slot.style.display = _activeRenderer ? '' : 'none';
    if (_activeRenderer && typeof _activeRenderer.reset === 'function') {
      try { _activeRenderer.reset(slot); } catch (e) { console.warn('[agentic-panels] reset failed', e); }
    }
  }

  if (!_activeRenderer) return false;
  if (typeof _activeRenderer.handle === 'function') {
    try {
      _activeRenderer.handle(slot, meta);
    } catch (e) {
      console.warn('[agentic-panels] handle failed', e);
    }
  }
  return true;
}

/** Side-load the built-in renderers. Called once from agentic.js. */
export async function loadBuiltins() {
  const { storybookRenderer } = await import('./storybook.js');
  register(storybookRenderer);
}
