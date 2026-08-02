/**
 * Files panel — public API shim.
 *
 * Real implementation lives in ./files/ (state, helpers, api, render,
 * preview, actions, index). Kept as a single file at this path so all
 * existing `import … from './files.js'` call sites keep working.
 */

export { initFiles, openFiles, closeFiles, toggleFiles } from './files/index.js';
