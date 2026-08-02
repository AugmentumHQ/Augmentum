"""augmentum-dev codebase model.

Public surface:
    from model import open_model, refresh
    db = open_model(project_root)
    refresh(db, project_root)
    db.execute("SELECT COUNT(*) FROM tables WHERE user_scoped = 1").fetchone()
"""

from __future__ import annotations

from .connection import find_project_root, open_model
from .ingest import refresh

__all__ = ["find_project_root", "open_model", "refresh"]
