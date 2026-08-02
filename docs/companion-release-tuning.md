# Companion — Release Tuning Checklist

This file tracks companion-kernel flags that have been flipped **ON for local
testing** and what their **safe public-release defaults** should be. Before a
public release, walk this list and revert anything marked "test-only" (or gate
it behind an explicit opt-in).

Design context: `~/.claude` memory `project_companion_kernel_telos` — the locked
foundation + 6-step build order. The governing invariant is that **none of these
flags may ever gate responsiveness**; they only shape what she does *unprompted*.
Enforced structurally by `tests/test_responsiveness_invariant.py`.

## Currently flipped ON for testing

| Flag | File | Test value | Safe public default | Notes |
|------|------|-----------|---------------------|-------|
| `companion_energy_enabled` | `config.py` | **True** (2026-06-20) | **False** | Step 2 of the kernel build. Energy damps OUTWARD (non-rest) autonomous activities when low → rest wins → she recovers (act→deplete→rest→recover duty cycle). Spends via the `spend_energy` verb on non-rest activities only. **Revert to False for release until tuned.** Runtime-toggleable via `config_routes` `_TOOL_SETTINGS`, so a release can also leave the default False and let instances opt in. |
| `companion_motion_cues_enabled` | `config.py` | **True** (2026-06-21) | **True (OK to ship on)** | NOT a test-only flip — a real feature. When her avatar is shown, the chat model may emit a hidden `[motion:xxx]` tag (stripped client-side by `motion-cue.js`) that drives an avatar gesture, mapped cue→roles through the user's curated/rated/uploadable clip pool. Low-risk (worst case = a stray stripped tag). Disable only if you don't want model-directed avatar animation. Runtime-toggleable via `config_routes`. |

## Already-on by default (NOT flipped by this work — context only)

| Flag | Default | Notes |
|------|---------|-------|
| `companion_tick_enabled` | `True` (config.py:832) | The autonomous tick loop (`behavior/tick.py`). Already on pre-this-work ("flipped after audit — utility-gated + exception-safe"). This is what makes energy *observable* — without it, no autonomous activity is chosen, so nothing spends energy. |

## Related flags left OFF (optional — flip if you want the fuller picture)

| Flag | Default | Notes |
|------|---------|-------|
| `companion_drives_enabled` | `False` | Drives (curiosity/competence/connection/rest) also modulate `activity_selector.choose()`. Independent of energy. Leaving it OFF means the current test isolates the **energy** effect cleanly. Flip ON to see drives + energy modulating together. |
| `companion_presence_mode` | `"silent"` | Gates the auxiliary autonomous writers (wondering, curator). NOT required for the energy duty cycle (the core `choose()` loop runs on `companion_tick_enabled`). Raise it if you want her ambient journaling/curation active too. |

## How to observe the energy gate (debug)

- **Logs** (`docker logs augmentum-augmentum-1`): `spend_energy_spent` fires on each
  non-rest autonomous activity; `tick_energy_decayed` shows the level recovering
  toward baseline.
- **DB**: `companion_energy_state.energy_level` per (user, companion). Baseline 0.6,
  floor 0.05.
- **Timescales** (current constants, `energy.py`): `SPEND_AMOUNT=0.10` per non-rest
  activity (so ~5–6 outward acts drain her from baseline toward the floor), decay
  half-life **6h** toward baseline (`companion_energy_decay_half_life_hours`). The
  spend is fast (minutes of activity); recovery is deliberately slow (hours). If you
  want faster recovery for a testing session, lower the half-life setting — it's
  validated in `config_routes` (range 0.5–168h).
