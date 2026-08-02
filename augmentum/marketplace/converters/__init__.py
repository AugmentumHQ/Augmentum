"""Universal store-format converters — foreign app manifests in, Augmentum
service listings out.

One converter per dialect, one shared eligibility gate:

- ``umbrel``  — getumbrel/umbrel-apps: umbrel-app.yml + docker-compose.yml
- ``runtipi`` — runtipi/runtipi-appstore: config.json + docker-compose.yml
  (schema v2, ``x-runtipi`` service extension)
- ``casaos``  — IceWhaleTech/CasaOS-AppStore AND bigbeartechworld/
  big-bear-casaos: single docker-compose.yml with ``x-casaos`` blocks

Every converter returns a :class:`ConversionResult`: an eligibility
verdict with concrete reasons (multi-container, host network, docker
socket, unpinned image, …), a candidate listing in our catalog schema,
and ``review`` flags for fields the source format simply doesn't carry
(the browser block above all) — the converter never silently invents
trust-relevant facts.

Used by the curation pipeline (scripts/curate_from_stores.py) today and
designed as the ingestion seam for community stores speaking foreign
formats tomorrow.
"""
from __future__ import annotations

from augmentum.marketplace.converters.base import ConversionResult
from augmentum.marketplace.converters.casaos import convert_casaos
from augmentum.marketplace.converters.runtipi import convert_runtipi
from augmentum.marketplace.converters.umbrel import convert_umbrel

__all__ = [
    "ConversionResult",
    "convert_casaos",
    "convert_runtipi",
    "convert_umbrel",
]
