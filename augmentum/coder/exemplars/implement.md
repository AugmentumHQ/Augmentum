# Implement exemplar

The shape of an implement turn: understand the existing pattern, change
all layers in lockstep, verify the change took effect via the project's
own validator. Applies to features that touch multiple files (settings,
routes, schemas, providers).

## User goal

"Add a setting `dream_summary_enabled` (boolean, default false) so users
can toggle dream summarization."

## Turn

[shell_exec "python .claude/skills/augmentum-dev/scripts/scaffold_setting.py dream_summary_enabled bool false"]
  # The project ships a scaffold. Used first — saves reading 4 files to
  # derive the same boilerplate by hand. The script PRINTS snippets for
  # each layer but does not apply them.
  → Layer 1 (config.py): dream_summary_enabled: bool = False
  → Layer 2 (config_routes.py _TOOL_SETTINGS): "dream_summary_enabled": (bool, 0, 1)
  → Layer 3 (server.py _SETTINGS_RESTORE_MAP): "dream_summary_enabled": _parse_bool
  → Layer 4 (settings.js): camelCase = dreamSummaryEnabled, 3 sites

[code_edit augmentum/config.py]
  + dream_summary_enabled: bool = False

[code_edit augmentum/proxy/config_routes.py]
  + "dream_summary_enabled": (bool, 0, 1),

[code_edit augmentum/proxy/server.py]
  + "dream_summary_enabled": _parse_bool,

[code_edit_batch ui/scripts/settings.js — 3 edits in one file]
  + dreamSummaryEnabled: false             (in DEFAULTS)
  + settings.dreamSummaryEnabled = data.dream_summary_enabled
                                           (in loadToolSettingsFromBackend)
  + dream_summary_enabled: settings.dreamSummaryEnabled
                                           (in syncToolSettingsToBackend)

[shell_exec "python .claude/skills/augmentum-dev/scripts/validate_wiring.py"]
  → All 4 layers present for dream_summary_enabled. No findings.

[finish_task]

## What this exemplar teaches

- Look for a project-native scaffold before writing boilerplate by hand.
  The codebase often ships its own; ignoring it is wasted iteration.
- Multi-file changes that share a contract (the 4-layer settings
  pattern, the route/handler/schema/test pattern) must move in lockstep.
  Don't ship a half-wired feature.
- Use `code_edit_batch` when multiple atomic edits target the same file.
  Use plain `code_edit` when files are independent.
- Verify with the project's own validator. For Augmentum settings:
  `validate_wiring.py` catches missing layers. Run it before claiming done.
- Pick the oracle to match the claim, before editing. A settings change
  has a wiring validator (used here). A new route claims "correct status +
  sane error" → contract/error tests. A UI change claims "the user can do
  X" → browser probe with one real interaction and a clean console, not a
  screenshot eyeball. The cheapest check that would FAIL if you were
  wrong is the right one.
- Stop with `finish_task` once the validator passes. The scanner's silence
  is the evidence; no celebration prose.
