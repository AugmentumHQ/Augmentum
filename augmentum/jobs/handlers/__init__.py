"""Registered background-job handlers.

Each handler module exposes a ``make_<name>_handler(app)`` factory that
returns the async handler bound to runtime services on ``app.state``
(http client, file index, data dir, etc.). The server startup sequence
calls these factories once and passes the result to
``register_handler(job_type, ...)``.
"""
