# Coder Workspace File Management

Status: **implemented, uncommitted** (2026-07-02). Not yet live-verified in a
running container. Two sprints closed the gap between the coder file panel and
a real IDE file explorer, driven by the observation that smaller models write
scripts fluidly, which shifts curation work onto the user — so the *curation
tools* had to get good.

A recurring pattern across both sprints: several gaps were **orphaned
endpoints or data** (semantic search, single-file download, turn-snapshot
diffs) — the backend already existed; only the UI surface was missing. Those
were the cheapest, highest-value wins.

---

## Sprint 1 — search, per-file diff, move

### Workspace search (text + semantic)
- **`augmentum/coder/text_search.py`** — structured text search over the live
  working tree. Runs `rg --json` inside the workspace container (the same
  isolation spine the agent's `code_grep` tool uses), falling back to `grep`
  on images without ripgrep. Returns per-match `{path, line, text, spans}`
  with byte→char offset conversion, long-line windowing, **leading-indent
  stripping** (indented matches were rendering as blank rows), and an honest
  `truncated` flag. Never silently caps.
- **`GET /api/coder/workspaces/{id}/search-text`** — `q`, `regex`, `case`,
  `glob`, `limit`. The semantic leg reuses the previously-UI-less
  **`GET /api/coder/search/{id}`** (codebase index).
- **`ui/scripts/coder-search.js`** — self-contained pane (callbacks for the
  active workspace id + open-file-at-line + build-index). Two legs
  (Text / Semantic), debounced with stale-response dropping, results grouped
  by file with highlighted spans, click opens the editor **at the line**.
  Toggle button in the Files header + **Ctrl+Shift+F**. Match rows carry a
  full-line `title` tooltip so a still-clipped trace is hover-readable.

### Per-file diff
- The reviewable-turn flow (`coder-review.js`, fired on `onReviewPending`)
  already surfaces *per-turn* diffs, so that was **not** duplicated. The
  missing piece was **on-demand** diff: right-click a changed file →
  **View changes** → modal reusing the commit panel's `_renderDiffHtml`.
- **`_fetchFileDiffSegments`** was extracted so the commit panel and the
  standalone modal share one fetch spine. Untracked (new) files render as a
  full addition — exactly what you want to review a freshly-sprayed script.

### Move
- Drag-to-move in the tree (custom `application/x-coder-path` drag type,
  disambiguated from the external-upload `Files` drag; folders and the tree
  root are drop targets), a **Move to…** prompt fallback, and a backend
  **destination-exists guard** (`rename` gained `overwrite`; returns 409 →
  confirm → retry).
- **Fixed a class bug:** rename used to *close* open editor tabs.
  `_retargetEditorPaths` now remaps open tabs (files and descendants of a
  moved folder) in place, preserving unsaved edits. This required making the
  editor's save read the live `_activeFilePath` rather than a captured
  closure, or a moved file would save back to its old path.
- `cm-editor.setCursor` now scrolls the target into view (search jumps were
  previously invisible).

### Left-panel resize (follow-up)
- Drag-to-resize grip on the left panel's right edge, mirroring the inspector
  panel's resize. Drives the (now-dynamic) `--panel-width` grid column;
  min 240 / max 60vw; persisted to `augmentum-panel-width`; double-click
  resets; ≥768px only. Lets long search traces breathe.

---

## Sprint 2 — download, multi-select, trash/undo, history

### Single-file download
- Wired the orphaned **`GET /api/coder/files/{id}/download`** (binary-safe) to
  a **Download** context-menu item.

### Multi-select + bulk operations
- Ctrl/Cmd-click toggles a row; Shift-click selects a range across visible
  rows. A selection surfaces a floating **`#coder-selection-bar`**
  (Move / Download / Delete / Clear).
- Plain click, Esc, or a workspace switch clears the selection —
  single-click-to-open is untouched.
- **`_moveMany`** is shared by the bulk-move button and multi-drag; dragging a
  selected row moves the whole selection (`{multi:[...]}` payload).
- Bulk delete collects all trash tokens into **one "Undo all"** toast; bulk
  move skips conflicts and reports a summary instead of blocking on the first.

### Soft-delete + undo (trash)
- **`ContainerManager.file_trash`** moves a path to
  `/workspace/.augmentum/trash/<id>/` with a manifest, and idempotently adds
  `.augmentum/trash/` to `.git/info/exclude` so trash never pollutes git
  status (`.augmentum` itself is tracked, so this matters).
- **`file_restore`** (conflict-guarded — never clobbers an occupied original),
  **`file_list_trash`**, **`file_purge_trash`**.
- **`DELETE /api/coder/files/{id}`** gains `permanent` (default false = trash →
  returns `trash_id` → **Undo** toast via `showToast` `opts.action`). New
  **`POST …/restore`**, **`GET …/trash`**, **`DELETE …/trash`**.
- **Isolation preserved:** the agent's own `file_delete` tool still hard-`rm`s;
  only the *UI route* soft-deletes. (Precedent: the separate `/api/files/*`
  subsystem already had trash/restore/bulk-delete.)

### Commit-history browser
- **`ContainerManager.git_show`** + **`GET /api/coder/checkpoints/{id}/show?hash=`**
  (hash regex-gated; splits the commit's metadata header from the diff body).
- The checkpoints list is now **browsable**: click a checkpoint to see *what
  changed in that commit* (`_showCommitDiffModal`, reuses the diff overlay +
  renderer). The existing Revert button is unchanged.

---

## Endpoint reference (added / newly-wired)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/coder/workspaces/{id}/search-text` | live text/regex search (structured) |
| GET | `/api/coder/search/{id}` | semantic index search (was UI-less) |
| GET | `/api/coder/files/{id}/download` | single-file download (was UI-less) |
| POST | `/api/coder/files/{id}/rename` | now takes `overwrite`; 409 on conflict |
| DELETE | `/api/coder/files/{id}` | now soft-deletes (`permanent` to hard-rm) |
| POST | `/api/coder/files/{id}/restore` | restore a trashed item |
| GET | `/api/coder/files/{id}/trash` | list trash (newest first) |
| DELETE | `/api/coder/files/{id}/trash` | purge one / empty trash |
| GET | `/api/coder/checkpoints/{id}/show?hash=` | a single commit's own diff |

## Key files

- `augmentum/coder/text_search.py` — search spine (new)
- `augmentum/coder/containers.py` — `file_trash`/`file_restore`/`file_list_trash`/`file_purge_trash`/`git_show`
- `augmentum/proxy/coder_routes.py` — routes
- `ui/scripts/coder-search.js` — search pane (new)
- `ui/scripts/coder.js` — tree, context menus, multi-select, move, diff/history modals, trash undo
- `ui/scripts/app.js` — `initLeftPanelResize`
- `ui/scripts/cm-editor.js` — `setCursor` scroll-into-view
- `ui/styles/coder.css`, `ui/styles/layout.css`, `ui/index.html`

## Tests

- `tests/test_coder_text_search.py` (18) — rg parse, offset conversion,
  windowing, indent-strip, truncation, grep fallback, search/rename routes.
- `tests/test_coder_trash_history.py` (12) — trash script build, manifest
  parse, restore conflict-guard, trash-list sort, delete/restore/show routes.

Two pre-existing route failures (`test_active_run_falls_back_to_ledger…`,
`test_stream_route_replays_ledger…`) reproduce at clean HEAD and are unrelated.

## Sprint 3 — familiar & seamless (ergonomics)

Frontend-only (`ui/scripts/coder.js`, `ui/styles/coder.css`). Closes the
biggest "this doesn't feel like an IDE" gaps.

### Inline rename & create (no more `prompt()`)
- **`_beginInlineRename`** edits the name in place on the row (Enter/blur
  commits, Esc cancels; files pre-select the base name, not the extension).
- **`_beginInlineCreate`** inserts an editable placeholder row — at the tree
  root or inside a folder (expanding it first). Wired to the New file / New
  folder header buttons and the folder context menu. The browser `prompt()`
  dialogs for rename/create are gone. (The command palette keeps its prompt
  path, since it can fire when the tree isn't visible.)

### Editor tabs — real multi-file editing
- Each tab now **caches its buffer + cursor** (`_snapshotActiveEditorBuffer`),
  so switching tabs no longer reloads from disk and **silently loses unsaved
  edits** — the previous behavior. Switching back restores the buffer and
  cursor.
- **Dirty indicator**: a dot on modified tabs that becomes the × close glyph
  on hover (`_markActiveDirty`, re-renders only on the clean→dirty edge).
- **Close guard**: confirm before closing a dirty tab; **workspace-switch
  guard**: confirm before discarding unsaved buffers (dropdown reverts on
  cancel). Save clears dirty + refreshes the buffer baseline.

### Reveal & follow the active file
- **`_revealInTree`** (called on every file open) expands the file's ancestor
  folders top-down (lazy-loading each via `_waitForChildren`), highlights the
  row, and scrolls it into view. Cheap when already visible. Makes
  "open from search" and tab switches show *where* the file lives.

### File size in tree rows
- `file_list` already returned `size`; rows now show a muted, right-aligned
  human-readable size (`_formatFileSize`), hidden during inline edit.

## Still open (smaller tail)

Keyboard navigation in the tree (↑↓/Enter/F2/Delete), merge-conflict
resolution UI, a `.env`/secrets editor, recent/pinned files, mtime in rows.
