"""Background-job runtime: handler registry + worker loop.

Consumers register an async handler under a ``job_type`` string, then
enqueue work through the store. The JobRunner picks up pending rows and
dispatches to the registered handler with a JobContext the handler uses
for progress reporting and cancel checks.

See ``augmentum/jobs/runner.py`` for the worker loop and
``augmentum/jobs/context.py`` for the handler-facing API.
"""

from __future__ import annotations

from augmentum.jobs.context import JobCancelled, JobContext
from augmentum.jobs.runner import JobHandler, JobRunner, register_handler

__all__ = [
    "JobCancelled",
    "JobContext",
    "JobHandler",
    "JobRunner",
    "register_handler",
]
