"""Generate curated service listings for data/marketplace/listings.json.

One-shot generator for the 2026-07-18 Umbrel-parity catalog expansion.
Source data: each app's umbrel-apps docker-compose.yml + umbrel-app.yml
(fetched 2026-07-18) — images are the exact pinned tags from those files.
Icons/screenshots reference the public umbrel-apps-gallery pages
(proxied client-side through /api/browse/image).

Run:  python scripts/gen_marketplace_listings.py
Idempotent: rebuilds the generated entries, preserves hand-authored ones.
"""
from __future__ import annotations

import json
from pathlib import Path

GALLERY = "https://getumbrel.github.io/umbrel-apps-gallery"

# Hand-authored ids that must never be overwritten by this generator.
PRESERVE = {"mkt:uptime-kuma", "mkt:navidrome", "mkt:radicale"}


def app(
    slug, title, category, tags, tagline, desc, image, port, *,
    env=None, prompts=None, cmd=None, vols=None, media="", hc="/", hct=90,
    after="setup_page", bpath="/", creds="user_set",
    ram=256, disk=300, dev="", site="", lic="", src="", gid=None, gext="jpg",
):
    gid = gid or slug
    service = {
        "id": slug,
        "name": title,
        "image": image,
        "port": port,
        "volumes": vols or {"data": "/data"},
        "healthcheck": {"path": hc, "timeout_s": hct},
    }
    if env:
        service["env"] = env
    if prompts:
        service["env_prompts"] = prompts
    if cmd:
        service["command"] = cmd
    if media:
        service["media_mount"] = media
    meta = {}
    if dev:
        meta["developer"] = dev
    if site:
        meta["website"] = site
    if lic:
        meta["license"] = lic
    # Screenshot extensions vary per gallery dir (verified live 2026-07-18:
    # most .jpg; grocy/wallos/changedetection-io ship .webp).
    meta["gallery"] = [f"{GALLERY}/{gid}/{n}.{gext}" for n in (1, 2, 3)]
    return {
        "id": f"mkt:{slug}",
        "title": title,
        "kind": "service",
        "publisher": "augmentum",
        "tagline": tagline,
        "description": desc,
        "thumbnail_url": f"{GALLERY}/{gid}/icon.svg",
        "source_url": src,
        "metadata": meta,
        "install_via": "service_manifest",
        "install_payload": {
            "manifest_version": 1,
            "service": service,
            "browser": {"after_install": after, "path": bpath, "credentials": creds},
            "resources": {"ram_mb": ram, "disk_mb": disk},
            "lifecycle": {"backup_paths": sorted((vols or {"data": "/data"}).values())},
        },
        "category": category,
        "tags": tags,
    }


APPS = [
    # ── Files & Productivity ─────────────────────────────────────────
    app("syncthing", "Syncthing", "files", ["sync", "files", "p2p"],
        "Continuous file sync between your devices, no cloud in between.",
        "Keeps folders in sync across your computers and phones, peer to peer and "
        "encrypted. First launch opens the web UI where you set a GUI password and add "
        "devices. Direct transfers use port 22000 when reachable; otherwise Syncthing "
        "falls back to community relays, which is slower but works everywhere.",
        "syncthing/syncthing:2.1.2", 8384,
        vols={"data": "/var/syncthing"}, ram=256, disk=200,
        dev="The Syncthing Foundation", site="https://syncthing.net",
        lic="MPL-2.0", src="https://github.com/syncthing/syncthing"),
    app("vaultwarden", "Vaultwarden", "files", ["passwords", "security", "bitwarden"],
        "Your own Bitwarden-compatible password vault.",
        "A lightweight server for the official Bitwarden apps and browser extensions. "
        "Everything is end-to-end encrypted with your master password. After install, "
        "create your account on the web vault, then point your Bitwarden clients at "
        "this server's HTTPS address.",
        "vaultwarden/server:1.36.0", 8089,
        env={"ROCKET_PORT": "8089"}, hc="/alive",
        ram=256, disk=300, dev="Daniel García & contributors",
        site="https://github.com/dani-garcia/vaultwarden",
        lic="AGPL-3.0", src="https://github.com/dani-garcia/vaultwarden"),
    app("stirling-pdf", "Stirling PDF", "files", ["pdf", "documents", "tools"],
        "Merge, split, rotate, convert, and edit PDFs in your browser.",
        "A complete PDF toolbox that runs entirely on your box — nothing you upload "
        "leaves it. Includes OCR in many languages, compression, page reordering, "
        "conversions, and more. Opens ready to use, no account needed.",
        "stirlingtools/stirling-pdf:2.14.2", 8080,
        env={"DOCKER_ENABLE_SECURITY": "false", "LANGS": "ALL"},
        vols={"tessdata": "/usr/share/tessdata", "configs": "/configs",
              "logs": "/logs", "pipeline": "/pipeline"},
        after="status", creds="none", hct=180, ram=1024, disk=800,
        dev="Stirling Tools", site="https://stirlingtools.com",
        lic="MIT", src="https://github.com/Stirling-Tools/Stirling-PDF"),
    app("memos", "Memos", "files", ["notes", "markdown", "journal"],
        "Capture quick thoughts in a lightweight, private notes stream.",
        "A privacy-first note-taking service with markdown, tags, and a timeline view. "
        "The first account you create becomes the admin.",
        "neosmemo/memos:0.29.1", 5230,
        env={"MEMOS_MODE": "prod"}, vols={"data": "/var/opt/memos"},
        ram=128, disk=200, dev="usememos", site="https://www.usememos.com",
        lic="MIT", src="https://github.com/usememos/memos"),
    app("flatnotes", "flatnotes", "files", ["notes", "markdown"],
        "Database-less markdown notes — every note is a plain file.",
        "A clean note-taking app where each note lives as a Markdown file you can back "
        "up, sync, or edit with anything. You choose the admin password during install; "
        "sign in as 'admin'.",
        "dullage/flatnotes:v5.5.4", 8080,
        env={"FLATNOTES_AUTH_TYPE": "password", "FLATNOTES_USERNAME": "admin"},
        prompts=[
            {"key": "FLATNOTES_PASSWORD", "label": "Password for the admin account", "secret": True},
            # Machine secret — auto-generated so the user never has to invent
            # "a long random string" (flatnotes exits 1 without it).
            {"key": "FLATNOTES_SECRET_KEY", "label": "Session secret (auto-generated)",
             "secret": True, "generate": True},
        ],
        after="login", ram=128, disk=100,
        dev="Adam Dullage", site="https://github.com/dullage/flatnotes",
        lic="MIT", src="https://github.com/dullage/flatnotes"),
    app("trilium", "Trilium Notes", "files", ["notes", "knowledge-base", "wiki"],
        "A personal knowledge base with an infinitely deep note tree.",
        "Build a structured knowledge base: hierarchical notes, cloning, relations, "
        "scripting, and encryption for protected notes. First launch walks you through "
        "setup and password creation.",
        "triliumnext/trilium:v0.103.0", 8080,
        env={"TRILIUM_DATA_DIR": "/data"}, gid="trilium-notes",
        ram=512, disk=400, dev="TriliumNext", site="https://github.com/TriliumNext",
        lic="AGPL-3.0", src="https://github.com/TriliumNext/Trilium"),
    app("pingvin-share", "Pingvin Share", "files", ["file-sharing", "uploads"],
        "Share files with anyone via links — a self-hosted WeTransfer.",
        "Upload files and get shareable links with optional expiry and passwords. The "
        "first account you register becomes the admin. Sharing outside your network "
        "requires this server to be reachable from outside.",
        "stonith404/pingvin-share:v1.13.0", 3000,
        vols={"data": "/opt/app/backend/data", "images": "/opt/app/frontend/public/img"},
        ram=384, disk=300, dev="Elias Schneider",
        site="https://stonith404.github.io/pingvin-share/introduction",
        src="https://github.com/stonith404/pingvin-share"),
    app("duplicati", "Duplicati", "files", ["backup", "encryption", "cloud"],
        "Encrypted backups to any storage you already have.",
        "Back up to S3, Backblaze, Google Drive, OneDrive, SFTP, WebDAV, and more — "
        "always encrypted before it leaves your box. You set the web console password "
        "during install. Optionally point it at a library folder to include in backups.",
        "duplicati/duplicati:2.3.0.4", 8200,
        env={"DUPLICATI__WEBSERVICE_PORT": "8200",
             "DUPLICATI__WEBSERVICE_INTERFACE": "any",
             "DUPLICATI__WEBSERVICE_ALLOWED_HOSTNAMES": "*"},
        prompts=[{"key": "DUPLICATI__WEBSERVICE_PASSWORD",
                  "label": "Web console password", "secret": True}],
        media="/source", after="login", ram=512, disk=300,
        dev="Duplicati team", site="https://duplicati.com",
        src="https://github.com/duplicati/duplicati"),
    app("mealie", "Mealie", "files", ["recipes", "meal-planning", "household"],
        "Recipe manager and meal planner — import recipes from any URL.",
        "Paste a recipe link and Mealie imports the ingredients and steps. Plan meals, "
        "build shopping lists, and share with your household. Create your account on "
        "first launch.",
        "ghcr.io/mealie-recipes/mealie:v3.20.1", 9000,
        env={"ALLOW_SIGNUP": "true", "MAX_WORKERS": "1", "WEB_CONCURRENCY": "1"},
        vols={"data": "/app/data"}, ram=512, disk=300,
        dev="Mealie contributors", site="https://docs.mealie.io",
        lic="AGPL-3.0", src="https://github.com/mealie-recipes/mealie"),
    app("grocy", "Grocy", "files", ["groceries", "household", "inventory"],
        "Groceries, chores, and household management — beyond your fridge.",
        "Track what's in stock, what's expiring, chores, and shopping lists. Sign in "
        "with the default account (admin / admin) and change the password right away.",
        "linuxserver/grocy:4.6.0", 80,
        env={"PUID": "1000", "PGID": "1000", "TZ": "Etc/UTC"},
        vols={"config": "/config"}, after="login", ram=256, disk=200,
        dev="Bernd Bestel", site="https://grocy.info", gext="webp",
        src="https://github.com/grocy/grocy"),
    app("homebox", "HomeBox", "files", ["inventory", "organization", "home"],
        "Inventory for your home — know what you own and where it is.",
        "Catalog your belongings with labels, locations, warranty details, and "
        "attachments. Register your account on first launch.",
        "ghcr.io/sysadminsmedia/homebox:0.26.2", 7745,
        # homebox hard-requires a >=32-byte API-key pepper or it panics on
        # boot; auto-generate it so install stays zero-config.
        prompts=[{"key": "HBOX_AUTH_API_KEY_PEPPER",
                  "label": "API key signing secret (auto-generated)",
                  "secret": True, "generate": True}],
        ram=128, disk=100, dev="SysAdmins Media", site="https://homebox.software",
        lic="AGPL-3.0", src="https://github.com/sysadminsmedia/homebox"),
    app("heimdall", "Heimdall", "files", ["dashboard", "launcher", "homepage"],
        "One tidy start page for all your self-hosted services.",
        "A simple, good-looking dashboard of links to everything you run. Opens ready "
        "to use — add your apps and links straight away.",
        "linuxserver/heimdall:2.8.1", 80,
        env={"PUID": "1000", "PGID": "1000"},
        vols={"config": "/config"}, after="status", creds="none",
        ram=256, disk=150, dev="LinuxServer.io", site="https://heimdall.site",
        src="https://github.com/linuxserver/Heimdall"),
    app("wallos", "Wallos", "files", ["subscriptions", "finance", "budget"],
        "Track recurring subscriptions and see what they really cost.",
        "Log your subscriptions, get renewal reminders, and see monthly and yearly "
        "totals by category. Register your account on first launch.",
        "bellamy/wallos:5.2.0", 80,
        env={"TZ": "Etc/UTC"},
        vols={"db": "/var/www/html/db",
              "logos": "/var/www/html/images/uploads/logos"},
        ram=256, disk=150, dev="Wallos", site="https://wallosapp.com",
        gext="webp", src="https://github.com/ellite/Wallos"),
    # ── Media ────────────────────────────────────────────────────────
    app("calibre-web", "Calibre-Web", "media", ["ebooks", "books", "reading"],
        "A clean web library for browsing and reading your eBooks.",
        "Browse, read, and download your Calibre eBook library from any browser. Sign "
        "in with the default account (admin / admin123), change the password, then "
        "point it at your books folder.",
        "linuxserver/calibre-web:0.6.26", 8083,
        env={"PUID": "1000", "PGID": "1000"},
        vols={"config": "/config"}, media="/books", after="login",
        ram=256, disk=200, dev="Janeczku & contributors",
        lic="GPL-3.0", src="https://github.com/janeczku/calibre-web"),
    app("metube", "MeTube", "media", ["youtube", "downloads", "video"],
        "Download videos from YouTube and hundreds of other sites.",
        "A friendly web UI for yt-dlp: paste a link, pick a format, and the download "
        "lands in its library folder. Opens ready to use.",
        "ghcr.io/alexta69/metube:2026.07.13", 8081,
        vols={"downloads": "/downloads"}, after="status", creds="none",
        ram=256, disk=200, dev="Alex Shnitman",
        src="https://github.com/alexta69/metube"),
    app("pinchflat", "Pinchflat", "media", ["youtube", "archiving", "automation"],
        "Automatically archive YouTube channels and playlists.",
        "Subscribe to channels or playlists and Pinchflat keeps local copies as new "
        "videos land — built on yt-dlp, designed for media-server layouts. Opens ready "
        "to use; you can add a password in its settings.",
        "ghcr.io/kieraneglin/pinchflat:v2025.6.6", 8945,
        vols={"config": "/config", "downloads": "/downloads"},
        after="status", creds="none", ram=512, disk=300,
        dev="Kieran Eglin", src="https://github.com/kieraneglin/pinchflat"),
    app("freshrss", "FreshRSS", "media", ["rss", "news", "reading"],
        "Follow every site you care about from one fast RSS reader.",
        "A free, self-hosted feed aggregator with keyboard-friendly reading, filters, "
        "and mobile client support. First launch runs a short setup wizard.",
        "linuxserver/freshrss:1.29.1", 80,
        env={"PUID": "1000", "PGID": "1000"},
        vols={"config": "/config"}, ram=256, disk=200,
        dev="FreshRSS community", site="https://freshrss.org",
        lic="AGPL-3.0", src="https://github.com/FreshRSS/FreshRSS"),
    # ── Networking ───────────────────────────────────────────────────
    app("librespeed", "LibreSpeed", "networking", ["speedtest", "network"],
        "Test connection speed between your devices and this server.",
        "A lightweight, no-tracking speed test served from your own box — useful for "
        "checking your LAN and Wi-Fi, not just your internet. Opens ready to use.",
        "ghcr.io/librespeed/speedtest:6.1.0", 8080,
        env={"PUID": "1000", "PGID": "1000"},
        vols={"config": "/config"}, after="status", creds="none",
        ram=128, disk=100, dev="LibreSpeed", site="https://librespeed.org",
        src="https://github.com/librespeed/speedtest"),
    app("myspeed", "MySpeed", "networking", ["speedtest", "monitoring", "history"],
        "Automatic speed tests with 30 days of history.",
        "Runs scheduled internet speed tests and charts the results so you can spot "
        "slowdowns and hold your ISP to its promises. Opens ready to use; a password "
        "can be set in its settings.",
        "germannewsmaker/myspeed:1.0.9", 5216,
        vols={"data": "/myspeed/data"}, after="status", creds="none",
        ram=256, disk=150, dev="Mathias Wagner", site="https://myspeed.dev",
        lic="MIT", src="https://github.com/gnmyt/myspeed"),
    # ── Automation ───────────────────────────────────────────────────
    app("changedetection", "changedetection.io", "automation",
        ["monitoring", "alerts", "web"],
        "Watch any web page and get notified when it changes.",
        "Track prices, restocks, job posts, or any page content, with flexible "
        "filters and dozens of notification targets. Opens ready to use; a password "
        "can be set in its settings.",
        "ghcr.io/dgtlmoon/changedetection.io:0.55.8", 5000,
        env={"HIDE_REFERER": "true", "DISABLE_VERSION_CHECK": "true"},
        vols={"datastore": "/datastore"}, after="status", creds="none",
        gid="changedetection-io", gext="webp", ram=256, disk=200,
        dev="dgtlmoon", site="https://changedetection.io",
        lic="Apache-2.0", src="https://github.com/dgtlmoon/changedetection.io"),
    app("n8n", "n8n", "automation", ["workflows", "integration", "no-code"],
        "Build automation workflows connecting hundreds of services.",
        "A visual workflow builder: triggers, branches, code steps, and integrations "
        "for hundreds of apps and APIs. Create the owner account on first launch.",
        "n8nio/n8n:2.30.5", 5678,
        env={"N8N_SECURE_COOKIE": "false", "N8N_DIAGNOSTICS_ENABLED": "false"},
        vols={"data": "/home/node/.n8n"}, ram=512, disk=400,
        dev="n8n GmbH", site="https://n8n.io",
        lic="Sustainable Use License (fair-code)", src="https://github.com/n8n-io/n8n"),
    app("node-red", "Node-RED", "automation", ["iot", "flows", "wiring"],
        "Wire together devices, APIs, and services with visual flows.",
        "Low-code programming for event-driven automation — drag nodes, wire them up, "
        "deploy. Opens ready to use; admin auth can be enabled in its settings file.",
        "nodered/node-red:5.0.1", 1880,
        vols={"data": "/data"}, after="status", creds="none",
        ram=384, disk=300, dev="OpenJS Foundation", site="https://nodered.org",
        lic="Apache-2.0", src="https://github.com/node-red/node-red"),
    # ── Developer Tools ──────────────────────────────────────────────
    app("code-server", "code-server", "developer", ["vscode", "editor", "ide"],
        "VS Code in your browser, running on your own box.",
        "The full VS Code experience served from this machine — consistent dev "
        "environment from any device. You set the editor password during install.",
        "codercom/code-server:4.128.0", 8080,
        prompts=[{"key": "PASSWORD", "label": "Editor password", "secret": True}],
        vols={"home": "/home/coder"}, after="login",
        ram=1024, disk=500, dev="Coder", site="https://coder.com",
        lic="MIT", src="https://github.com/coder/code-server"),
]


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "marketplace" / "listings.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    generated_ids = {a["id"] for a in APPS}
    kept = [entry for entry in doc["listings"]
            if entry["id"] in PRESERVE or entry["id"] not in generated_ids]
    doc["listings"] = kept + APPS
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"wrote {len(doc['listings'])} listings ({len(APPS)} generated)")


if __name__ == "__main__":
    main()
