"""Augmentum entry point."""

from __future__ import annotations

import uvicorn

from augmentum.config import settings
from augmentum.utils.logging import setup_logging


def main() -> None:
    """Start the Augmentum server."""
    setup_logging(settings.log_level)
    uvicorn.run(
        "augmentum.proxy.server:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        ws_max_size=settings.ws_max_frame_bytes,
    )


if __name__ == "__main__":
    main()
