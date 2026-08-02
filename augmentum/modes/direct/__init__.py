"""Direct mode — raw passthrough to the backend, no Augmentum injection.

Intended for external API-key clients (Claude Code, Aider, Cline, headless
coding tools) that want a zero-overhead path to the underlying model.
The route layer short-circuits before any injection sites; this handler
is intentionally trivial — see ``augmentum/modes/direct/handler.py``.
"""
from augmentum.modes.direct.handler import DirectHandler

__all__ = ["DirectHandler"]
