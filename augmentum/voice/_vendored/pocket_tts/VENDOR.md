# Vendored: kyutai-labs/pocket-tts (override-mode vendor)

This directory is Augmentum's pinned-version override of the
[`pocket-tts`](https://github.com/kyutai-labs/pocket-tts) Python package
from Kyutai Labs. It exists so we can capture **Mimi codec tokens** from
the model's internal generation loop — upstream's public API only exposes
the decoded PCM tensor, but Phase 2 of the presence pipeline needs the
token stream for forward-compatible audio history (per
`[[substrate-paying-back]]` and the Phase-2 design note in
`augmentum/companion/presence/audio_history.py`).

## Vendor strategy: pinned override, not wholesale copy

Two paths were considered:

| Strategy | Pros | Cons |
|----------|------|------|
| **Wholesale vendor** — copy `pocket_tts/` and all its 5k+ LOC of transitive deps into this tree | Fully owned code; survives any upstream rename or removal | 5k+ LOC of maintenance burden, license attribution per file, manual upgrade diff-merge, can't be exercised locally without re-installing every transitive dep |
| **Pinned override** (chosen) | Small surface (~150 LOC), tracks upstream cleanly, breakages surface loudly at import, simpler upgrades | Depends on upstream API shape staying recognizable across versions |

The override approach picks **one specific upstream commit** as the
contract, subclasses `TTSModel` to add the Mimi tap, and **fails loudly
at import** if the upstream API drifts (the tap.py module checks the
upstream version pin at construction time). This is materially safer
than a "best-effort" patch that silently no-ops when upstream changes.

If upstream ever moves `_decode_audio_worker` into something we can't
override cleanly, the documented fallback is to flip to wholesale vendor
of `tts_model.py` + its three direct deps (`mimi.py`, `flow_lm.py`,
`models/__init__.py`). That migration is one commit; the override
manifest below has the file list pre-staged.

## Upstream pin

| Field | Value |
|-------|-------|
| Repository | https://github.com/kyutai-labs/pocket-tts |
| Pinned commit | `15a6c1817b360f9b37691aef9734435a85610c68` |
| Pinned date | 2026-06-03 |
| Upstream license | MIT (full text in LICENSE.upstream) |
| PyPI package | `pocket-tts` |

The override module **reads this commit SHA at import** and compares it
against `pocket_tts.__version__`. A version mismatch logs a warning so
prod ops can see drift, but does not raise — the tap will still attempt
to attach. A drift event in the wild should trigger a code review against
the new upstream `tts_model.py` to confirm the override surface still
applies.

## Tap point

The tap subclasses `pocket_tts.models.tts_model.TTSModel` and overrides
`_decode_audio_worker` (line ~433 in pinned upstream). The override
captures the **quantized Mimi latent** right after `self.mimi.quantizer`
returns and before `self.mimi.decode_from_latent`. The quantized latent
is what feeds the vocoder; for the audio-history substrate we encode
it as int16 token indices via the quantizer's `codes_for_latent` helper
when available, otherwise we store the raw quantized tensor.

Codes are emitted via an optional `mimi_codes_callback` on the model
instance — set to a callable to receive `(chunk_index, codes_tensor)`
per audio frame, set to `None` to skip the tap entirely (zero overhead
when capture is disabled).

The tap is **deliberately additive**: the parent decode path runs
unchanged, the audio output stream is byte-identical to upstream. Only
the side channel changes. This minimizes the blast radius if the tap
itself has a bug.

## Files in this directory

| File | Purpose |
|------|---------|
| `VENDOR.md` | This document — the contract for what's vendored, why, and how to upgrade. |
| `LICENSE.upstream` | Verbatim copy of upstream MIT license per its "include in substantial portions" clause. |
| `__init__.py` | Public API surface — exports `MimiTappedTTSModel`, version pin constants, and the tap-installation helper. |
| `upstream_pin.py` | Constants only: `UPSTREAM_COMMIT`, `UPSTREAM_DATE`, `MIN_TESTED_VERSION`. Imported separately so a quick `python -c "from augmentum.voice._vendored.pocket_tts.upstream_pin import UPSTREAM_COMMIT; print(UPSTREAM_COMMIT)"` works without pulling in torch. |
| `tap.py` | The `MimiTappedTTSModel` subclass. Imports upstream lazily; raises a clear ImportError if `pocket_tts` isn't installed. |

## When to upgrade the pin

Upgrade the pin (and re-test the tap against the new upstream commit) when:

1. Augmentum's bundled image bumps `pocket-tts` to a newer release
2. Upstream ships a fix we need (check `Dockerfile.gpu` line 168)
3. Quarterly review: bump to current `main` HEAD and re-validate

The upgrade procedure:

1. Fetch new upstream `tts_model.py` and diff against the pinned commit
2. If `_decode_audio_worker`'s signature or body shape changed, update `tap.py`
3. Update `UPSTREAM_COMMIT` + `UPSTREAM_DATE` in `upstream_pin.py`
4. Run the smoke tests: they import the module + verify the public API contract
5. (Production check, requires the deps) integration test against the GPU image

## When to flip to wholesale vendor

The override approach is the right call until any of:

- Upstream `_decode_audio_worker` becomes private behind a property wall
  we can't reach
- Upstream becomes archived / unmaintained
- We need to patch >1 internal method (the surface gets too brittle)
- We need a feature upstream won't accept (e.g. a Mimi-codes-only fast
  path that bypasses the vocoder for token-only history)

The migration is well-scoped: vendor `pocket_tts/models/tts_model.py`,
`pocket_tts/models/mimi.py`, `pocket_tts/models/flow_lm.py`, and
`pocket_tts/models/__init__.py` (about 60KB total). The transitive deps
from `pocket_tts/modules/`, `pocket_tts/conditioners/`, and
`pocket_tts/utils/` can stay as upstream imports — only the
overridden-method-host files need to be ours.

## Why not just monkey-patch at runtime?

Monkey-patching the upstream class at `pocket_tts.py` import time would
work for one process but:

- Hides the patch from `grep` (looks like normal upstream code)
- Breaks when a second consumer imports the upstream model
- Can't be reasoned about without running the code

Subclassing in a vendored module makes the patch a named class with a
named override, discoverable by `grep MimiTappedTTSModel`, and testable
in isolation.
