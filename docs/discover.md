# Discover — one-click services, with Augmentum as the gateway

Beyond its own capabilities, Augmentum is a front door to a catalog of **~50
companion services** you can install with one click — media servers, file and
productivity tools, networking utilities, automation, and extra model backends.
Augmentum wires each one up so it's reachable through a single trusted gateway,
with **automatic HTTPS** and **mDNS discovery**, instead of juggling ports and
self-signed-certificate warnings.

## The catalog

Open the **Discover** panel. Services are grouped by category — for example:

- **Providers** — extra model/inference backends (e.g. a **vLLM model-swap**
  server).
- **Media** — Jellyfin, Emby, Audiobookshelf, and more.
- **Files & Productivity** — document, notes, and file tools.
- **Networking** — proxies and network utilities.
- **Automation** — schedulers and automation platforms.
- **Add-ons** — Augmentum-specific extensions.

(Examples include **Jellyfin**, **Suwayomi**, and a **vLLM-swap** backend — the
exact set grows over time.)

## Installing a service

1. Find a service in Discover and click **Install**.
2. Augmentum pulls its container, starts it, and puts it behind the gateway:
   - **Automatic HTTPS** via the bundled local certificate authority — no
     browser warnings, no Let's Encrypt setup.
   - **mDNS** so it's discoverable on your LAN.
   - Reachable through Augmentum's trusted origin.
3. Set a **memory limit** for the service if you want to cap its footprint, and
   **Update** it later from the same panel.

Because everything runs under Docker with Augmentum as the gateway, a service you
install is reachable the same way from your phone, TV, or another machine on the
network (and, with Tailscale, from anywhere) — without opening a port per app.

## Why route through Augmentum

- **One trusted origin** instead of a dozen `http://host:port` tabs with cert
  warnings.
- **Automatic TLS** for everything, from the same local CA your browser already
  trusts (one-time trust install per device).
- **Discoverable** on the LAN via mDNS.
- **Managed lifecycle** — install, memory-limit, and update from one panel.

## Custom stores

The catalog is extensible: you can add your own **store** (a source of listings)
from the Discover panel and sync it, so a team or community can share a curated
set of installable services beyond the built-in catalog.

## Using an installed service with Augmentum

Some services become capabilities Augmentum itself uses — a **vLLM-swap** backend
shows up as a model provider; a media server plugs into the media surfaces
(Jellyfin/Emby/Plex/Audiobookshelf). Others are just conveniently hosted behind
the same gateway. Either way, you install once and reach it everywhere.
