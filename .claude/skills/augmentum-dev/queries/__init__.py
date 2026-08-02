"""Codebase-model queries — diagnostic checks expressed as SQL.

Each query module exposes:
    NAME                — stable identifier (e.g. "orphaned_endpoints")
    DESCRIPTION         — one-line summary for audit output
    QUERY               — SQL returning rows of offenders
    DIAGNOSE (optional) — SQL returning aggregate breakdown rows
    SEVERITY (optional) — fn(count) -> "ok" | "warn" | "error"

The audit dispatcher discovers queries by importing every submodule
under this package and looking for the NAME symbol.
"""

from __future__ import annotations

from . import (
    incomplete_settings,
    multi_tenant_audit,
    orphaned_endpoints,
    subsystem_health,
    untested_routes,
)

ALL_QUERIES = [
    orphaned_endpoints,
    incomplete_settings,
    untested_routes,
    multi_tenant_audit,
    subsystem_health,  # purely informational; runs last
]

__all__ = [
    "ALL_QUERIES",
    "incomplete_settings",
    "multi_tenant_audit",
    "orphaned_endpoints",
    "subsystem_health",
    "untested_routes",
]
