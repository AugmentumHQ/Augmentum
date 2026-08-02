"""Docker network management for managed provider services.

Ensures all managed containers join the same Docker network as the
Augmentum compose stack so they can communicate via container name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiodocker

log = get_logger(__name__)

_NETWORK_PREFIX = "augmentum"
_NETWORK_SUFFIX = "default"


async def ensure_network(docker: aiodocker.Docker) -> str:
    """Find or create the augmentum Docker network.

    First tries to find the existing compose-created network
    (augmentum_default). If not found, creates one.

    Returns the network name.
    """
    networks = await docker.networks.list()
    for net in networks:
        name = net["Name"]
        if _NETWORK_PREFIX in name.lower() and _NETWORK_SUFFIX in name.lower():
            log.debug("network_found", name=name)
            return name

    network_name = f"{_NETWORK_PREFIX}_{_NETWORK_SUFFIX}"
    try:
        await docker.networks.create({
            "Name": network_name,
            "Driver": "bridge",
            "Labels": {"augmentum.managed": "true"},
        })
        log.info("network_created", name=network_name)
    except Exception:
        log.debug("network_create_skipped", name=network_name)
    return network_name


async def connect_container(
    docker: aiodocker.Docker,
    container_id: str,
    network_name: str,
    aliases: list[str] | None = None,
) -> None:
    """Connect a container to the augmentum network with optional aliases."""
    net = await docker.networks.get(network_name)
    endpoint_config = {}
    if aliases:
        endpoint_config["Aliases"] = aliases
    try:
        await net.connect({"Container": container_id, "EndpointConfig": endpoint_config})
        log.debug("container_connected", container=container_id[:12], network=network_name)
    except Exception as exc:
        if "already exists" in str(exc).lower():
            return
        raise
