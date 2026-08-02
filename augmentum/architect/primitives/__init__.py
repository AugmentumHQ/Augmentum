"""Architect-callable primitives.

Each module in this package registers one or more Action handlers via
``@register_action``. The package's parent __init__ imports the modules
so the decorators fire at process startup.

Adding a primitive:
  1. Create a new module here (e.g. ``image_defaults.py``).
  2. Define the handler + arg_inferrer.
  3. Call ``register_action(...)`` with ``surfaces`` set and
     ``arg_inferrer`` provided.
  4. Import the module from ``augmentum/architect/__init__.py``.
  5. Add a smoke test that imports the module + checks registration.
"""

from __future__ import annotations
