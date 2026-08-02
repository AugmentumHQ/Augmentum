-- Modular preset fields — toggle-driven composition + anti-slop phrase list.
-- Additive: existing presets keep working; new fields are nullable/empty.

ALTER TABLE prompt_presets ADD COLUMN modular_config TEXT NOT NULL DEFAULT '';
ALTER TABLE prompt_presets ADD COLUMN anti_slop_phrases TEXT NOT NULL DEFAULT '';

-- Seed the new Modular preset and promote it to default. Existing default
-- is demoted to is_default=0 (not deleted). The app's seed_builtins() step
-- inserts it on fresh installs; this migration handles upgrades.
UPDATE prompt_presets SET is_default = 0 WHERE is_default = 1;

INSERT OR IGNORE INTO prompt_presets (
    id, name, system_prompt, jailbreak, post_history, author_note,
    author_note_depth, is_default, modular_config, anti_slop_phrases,
    updated_at
) VALUES (
    'builtin_modular',
    'Modular',
    '',
    '',
    '',
    '',
    4,
    1,
    '{"role":"roleplayer","tense":"present","pov":"third","pov_mode":"character","length":"moderate","content":"sfw","anti_slop":true}',
    '',
    datetime('now')
);

-- Ensure the modular preset is the default, even if INSERT OR IGNORE skipped
-- because a row with that id already existed.
UPDATE prompt_presets SET is_default = 1 WHERE id = 'builtin_modular';
