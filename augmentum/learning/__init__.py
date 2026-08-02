"""Language-learning domain logic.

This package holds the *algorithmic* pieces of the language-learning
system — the FSRS spaced-repetition scheduler and the language-pack
lookup layer. Persistence lives in ``augmentum/state/vocab_store.py``;
HTTP routes in ``augmentum/proxy/learning_routes.py``.

See ``docs/superpowers/specs/2026-05-11-language-learning-system.md``.
"""

from __future__ import annotations
