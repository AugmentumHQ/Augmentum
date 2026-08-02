"""Abstract framework adapter — the contract every concrete adapter
satisfies.

The bug_finder treats codebases through an adapter so the same agent
loop (lead → investigator → detector → verifier → fixer) works on a
FastAPI project, a Flask app, an Express service, etc. The adapter
exposes:

* ``list_routes()`` — HTTP routes / endpoints
* ``list_jobs()`` — background jobs / tasks / queues
* ``list_settings_files()`` — config / settings locations
* ``identify_route_file(path)`` — is this file a router definition?
* ``identify_test_command()`` — how do you run this project's tests?

Adapters are pure-Python: they read structural facts from the
workspace, return typed records, and never invoke an LLM. The agent
asks the adapter "give me all the auth-shaped routes" instead of
asking the model to grep for ``@router.X`` decorators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterRouteHint:
    """One HTTP route / endpoint discovered by the adapter."""

    method: str             # "GET", "POST", "WEBSOCKET", ...
    path: str               # e.g. "/api/users/{id}"
    handler: str            # "file.py:function_name"
    file: str               # repo-relative source file
    line: int = 0           # 1-based line number when known


@dataclass(frozen=True)
class AdapterSettingHint:
    """One configuration / settings location discovered by the adapter."""

    file: str               # repo-relative source file
    kind: str               # "python_class" | "json" | "yaml" | "env" | ...
    name_hint: str = ""     # optional: name of the settings symbol


class FrameworkAdapter(ABC):
    """Common interface every framework adapter satisfies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework slug — ``"fastapi"``, ``"flask"``, ``"express"``, …"""

    @abstractmethod
    def list_routes(self, root: Path) -> list[AdapterRouteHint]:
        """Enumerate every HTTP route declared under ``root``.

        Implementations are best-effort — when a route can't be
        statically resolved (dynamic registration, plugin-style
        mounting), the adapter skips it rather than guessing. The
        agent gets the routes it CAN trust; the investigator handles
        the rest.
        """

    def list_jobs(self, root: Path) -> list[str]:
        """Enumerate background job / task / queue declarations.

        Default returns ``[]`` for adapters that don't ship job
        detection. Override on a per-framework basis (e.g. Celery
        ``@app.task``, Augmentum's ``register_job``).
        """
        return []

    def list_settings_files(self, root: Path) -> list[AdapterSettingHint]:
        """Files that look like config / settings definitions."""
        return []

    def identify_route_file(self, path: Path | str) -> bool:
        """``True`` when ``path`` is structurally a route file.

        Used by the investigator to decide whether grep-hits in a file
        are likely route-handler matches. Override per-framework.
        """
        return False

    def identify_test_command(self, root: Path) -> str:
        """Detected test command for the project (``""`` when unknown)."""
        return ""


class NullAdapter(FrameworkAdapter):
    """Fallback for unknown frameworks. Every query returns empty.

    Callers get to invoke methods without a None check; the agent
    sees "no routes / no jobs / no settings" and gracefully degrades
    to LLM exploration. This keeps the contract honest — we never
    guess at structure we can't actually verify.
    """

    @property
    def name(self) -> str:
        return "null"

    def list_routes(self, root: Path) -> list[AdapterRouteHint]:
        return []
