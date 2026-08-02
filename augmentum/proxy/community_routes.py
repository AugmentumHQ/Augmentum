"""Community install handler — receives /community-install from augmentumhq.com.

Spec: ``docs/specs/community-install.md`` in the augmentumhq-site repo.

Two endpoints:

* ``GET /community-install?manifest_url=...`` — public (auth middleware
  exempt). Self-redirects to ``/login?next=...`` if no session. With a
  session, fetches the manifest + artifact, renders a preview UI, and
  hands the user a confirm button.

* ``POST /api/community/install`` — auth-gated. Performs the actual
  import for the logged-in user. v0 dispatches characters and
  reasoning-flows; powers and knowledge return 501.

Every install writes a row to ``community_installs`` (migration 236) for
the audit trail. Every fetch goes through ``SafeHttpClient`` to block
loopback / RFC1918 / link-local targets. Every URL is checked against a
trusted-origin allowlist before fetching.
"""

from __future__ import annotations

import json
import uuid
from html import escape
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from augmentum.marketplace.install_dispatchers import (
    _install_character,
    _install_knowledge_pack,
    _install_power,
    _install_reasoning_flow,
)
from augmentum.proxy import character_routes as _char
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError

log = get_logger(__name__)

router = APIRouter(tags=["community"])

# Trusted community origins. Additional origins can be added via the
# admin setting ``community_trusted_origins``. The defaults below are
# the canonical AugmentumHQ/community repo on GitHub.
_BUILTIN_TRUSTED_ORIGINS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/AugmentumHQ/",
    "https://raw.githubusercontent.com/augmentumhq/",
)

_KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {"characters", "reasoning-flows", "powers", "knowledge"}
)

_REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "slug",
    "name",
    "category",
    "description",
    "version",
    "source_url",
)


# ── Helpers ───────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


async def _resolve_user(request: Request):
    """Resolve the authenticated user for a public-path route.

    AuthMiddleware short-circuits for paths in ``_PUBLIC_PATHS`` and
    never attaches ``scope["user"]``. For routes that *want* to know
    whether a session exists (this preview, /api/auth/status), we have
    to parse the token ourselves and validate it. Mirrors
    ``augmentum/proxy/auth_routes.py::auth_status``.
    """
    user = request.scope.get("user")
    if user:
        return user
    sm = getattr(request.app.state, "session_manager", None)
    if not sm:
        return None
    token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    else:
        for part in request.headers.get("cookie", "").split(";"):
            part = part.strip()
            if part.startswith("augmentum_session="):
                token = part[len("augmentum_session="):].strip()
                break
    if not token:
        return None
    try:
        return await sm.validate_token(token)
    except Exception:
        return None


def _settings(request: Request):
    return getattr(request.app.state, "settings", None)


def _trusted_origins(request: Request) -> tuple[str, ...]:
    settings = _settings(request)
    extra: list[str] = []
    if settings is not None:
        raw = getattr(settings, "community_trusted_origins", None) or []
        if isinstance(raw, list):
            extra = [str(s) for s in raw if isinstance(s, str)]
    return _BUILTIN_TRUSTED_ORIGINS + tuple(extra)


def _is_allowed_community_url(url: str, request: Request) -> bool:
    return any(url.startswith(prefix) for prefix in _trusted_origins(request))


def _community_enabled(request: Request) -> bool:
    settings = _settings(request)
    if settings is None:
        return True
    return bool(getattr(settings, "community_install_enabled", True))


def _http_status_error(meta: dict, label: str) -> str:
    raw_status = meta.get("status_code")
    if raw_status is None:
        return ""
    try:
        status = int(raw_status)
    except (TypeError, ValueError):
        return ""
    if 200 <= status < 300:
        return ""
    final_url = str(meta.get("url") or "").strip()
    if final_url:
        return f"{label} returned HTTP {status}: {final_url}"
    return f"{label} returned HTTP {status}."


# ── Per-category install dispatchers ──────────────────────────────────
# Moved to augmentum/marketplace/install_dispatchers.py so the Discover
# routes can share them. Imported at the top of this file; the
# definitions used to live below this comment.


# ── Audit row write ───────────────────────────────────────────────────


async def _record_install(
    request: Request,
    *,
    user_id: str,
    manifest_url: str,
    category: str,
    slug: str,
    item_version: str,
    installed_resource_id: str,
) -> None:
    be = _char._backend(request)
    if not be:
        log.warning("community_install_audit_skipped_no_backend")
        return

    try:
        await be.execute(
            """
            INSERT INTO community_installs (
                id, user_id, manifest_url, category, slug,
                item_version, installed_resource_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                user_id,
                manifest_url,
                category,
                slug,
                item_version,
                installed_resource_id,
            ),
        )
        await be.commit()
    except Exception as exc:  # audit row failure must not break install
        log.warning("community_install_audit_failed", error=str(exc))


# ── GET /community-install ────────────────────────────────────────────


@router.get("/community-install", response_model=None)
async def community_install_preview(request: Request, manifest_url: str = ""):
    """Render the install preview UI for a community item.

    Public route (auth middleware exempt). Renders the preview regardless
    of auth state — unauthenticated users see the preview *with* an inline
    login form (so they know what they're agreeing to install before
    signing in); authenticated users see the preview *with* a confirm
    button that POSTs to /api/community/install.

    Why no /login redirect: the SPA at `/` handles auth internally and
    doesn't honor a `?next=` query param after login. Redirecting there
    would lose the install context. Inlining the login keeps the user
    on the preview page, which both helps them trust what they're about
    to install and avoids a round-trip through the main UI.
    """
    if not _community_enabled(request):
        return _render_error_html(
            "Community install disabled",
            "An admin has disabled community installs on this Augmentum instance.",
        )

    if not manifest_url:
        return _render_error_html(
            "Missing parameter",
            "This page expects a manifest_url query parameter pointing at a "
            "community item's manifest.yaml file.",
        )

    if not _is_allowed_community_url(manifest_url, request):
        return _render_error_html(
            "Untrusted source",
            "The URL is not from a trusted community source. "
            "Trusted prefixes can be configured via admin settings.",
        )

    # Fetch + validate manifest
    try:
        client = SafeHttpClient(max_response_size=64 * 1024)
        manifest_text, _meta = await client.fetch(manifest_url, timeout=10.0)
    except SafeHttpError as exc:
        return _render_error_html("Couldn't fetch manifest", str(exc))
    if status_error := _http_status_error(_meta, "Manifest URL"):
        return _render_error_html("Couldn't fetch manifest", status_error)

    try:
        manifest = yaml.safe_load(manifest_text)
    except yaml.YAMLError as exc:
        return _render_error_html("Invalid manifest YAML", str(exc))

    if not isinstance(manifest, dict):
        return _render_error_html(
            "Invalid manifest", "Manifest must be a YAML mapping at top level."
        )

    missing = [f for f in _REQUIRED_MANIFEST_FIELDS if not manifest.get(f)]
    if missing:
        return _render_error_html(
            "Incomplete manifest",
            f"Manifest is missing required fields: {', '.join(missing)}",
        )

    category = str(manifest["category"])
    if category not in _KNOWN_CATEGORIES:
        return _render_error_html(
            "Unknown category", f"Category '{category}' is not recognized."
        )

    # Fetch + validate artifact
    source_url = str(manifest["source_url"])
    if not _is_allowed_community_url(source_url, request):
        return _render_error_html(
            "Untrusted artifact source",
            "The manifest's source_url is not from a trusted community source.",
        )

    try:
        artifact_client = SafeHttpClient(max_response_size=5 * 1024 * 1024)
        artifact_text, artifact_meta = await artifact_client.fetch(source_url, timeout=15.0)
    except SafeHttpError as exc:
        return _render_error_html("Couldn't fetch artifact", str(exc))
    if status_error := _http_status_error(artifact_meta, "Artifact URL"):
        return _render_error_html("Couldn't fetch artifact", status_error)

    try:
        if category == "powers":
            artifact: Any = artifact_text  # raw markdown
        else:
            artifact = json.loads(artifact_text)
    except json.JSONDecodeError as exc:
        return _render_error_html("Invalid artifact JSON", str(exc))

    # Render preview — with inline login form OR confirm button
    user = await _resolve_user(request)
    return HTMLResponse(
        _render_preview_html(manifest, artifact, category, manifest_url, user)
    )


# ── POST /api/community/install ───────────────────────────────────────


class CommunityInstallRequest(BaseModel):
    manifest_url: str = Field(..., min_length=1, max_length=2048)
    category: str = Field(..., min_length=1, max_length=32)
    artifact: dict | str
    manifest: dict | None = None


@router.post("/api/community/install")
async def community_install(request: Request) -> JSONResponse:
    """Perform the actual install for a community item.

    Auth-gated. The user_id from the session scopes the install — items
    land in the logged-in user's account only.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not _community_enabled(request):
        raise HTTPException(status_code=403, detail="Community install disabled")

    try:
        body_raw = await request.json()
        body = CommunityInstallRequest.model_validate(body_raw)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc

    if not _is_allowed_community_url(body.manifest_url, request):
        raise HTTPException(status_code=400, detail="Untrusted manifest URL")

    if body.category not in _KNOWN_CATEGORIES:
        raise HTTPException(
            status_code=400, detail=f"Unknown category: {body.category}"
        )

    if body.category == "characters":
        result_id = await _install_character(request, body.artifact, user_id)
    elif body.category == "reasoning-flows":
        result_id = await _install_reasoning_flow(request, body.artifact, user_id)
    elif body.category == "powers":
        result_id = await _install_power(request, body.artifact, user_id)
    elif body.category == "knowledge":
        result_id = await _install_knowledge_pack(request, body.artifact, user_id)
    else:  # defensive — caught above
        raise HTTPException(status_code=400, detail=f"Unhandled category: {body.category}")

    slug = str((body.manifest or {}).get("slug", "unknown"))
    version = str((body.manifest or {}).get("version", ""))
    await _record_install(
        request,
        user_id=user_id,
        manifest_url=body.manifest_url,
        category=body.category,
        slug=slug,
        item_version=version,
        installed_resource_id=result_id,
    )

    log.info(
        "community_item_installed",
        user_id=user_id,
        category=body.category,
        slug=slug,
        resource_id=result_id,
    )
    return JSONResponse({"status": "installed", "id": result_id})


# ── HTML rendering ────────────────────────────────────────────────────


def _render_error_html(title: str, detail: str) -> HTMLResponse:
    html = _BASE_PAGE.format(
        title=escape(title),
        body=(
            f'<div class="error-box">'
            f'<h1>{escape(title)}</h1>'
            f'<p>{escape(detail)}</p>'
            f'<p><a href="/">Back to Augmentum</a></p>'
            f'</div>'
        ),
    )
    return HTMLResponse(html, status_code=400)


def _render_preview_html(
    manifest: dict, artifact: Any, category: str, manifest_url: str, user=None
) -> str:
    cat_label = _category_label(category)
    name = escape(str(manifest.get("name", "Untitled")))
    description = escape(str(manifest.get("description", "")))
    author_raw = manifest.get("author") or {}
    if isinstance(author_raw, dict):
        author = escape(str(author_raw.get("name", "unknown")))
    else:
        author = escape(str(author_raw))
    version = escape(str(manifest.get("version", "")))
    min_version = escape(str(manifest.get("augmentum_min_version", "")))
    license_str = escape(str(manifest.get("license", "CC0")))
    tags = manifest.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tag_html = "".join(f'<span class="tag">{escape(str(t))}</span>' for t in tags)

    artifact_json = json.dumps(artifact, indent=2, default=str)
    if len(artifact_json) > 8000:
        artifact_json = artifact_json[:8000] + "\n...\n[truncated for preview]"

    if user is not None:
        auth_section = (
            f'<div class="actions">'
            f'<button type="button" class="btn primary" id="confirm-btn">'
            f'Install to my account'
            f'</button>'
            f'<a href="/" class="btn secondary">Cancel</a>'
            f'</div>'
            f'<p class="signed-in">Installing as <strong>{escape(user.username)}</strong>.</p>'
        )
    else:
        auth_section = (
            '<div class="login-box">'
            '<h2>Sign in to install</h2>'
            '<p>This item will be added to your account. Augmentum runs '
            'on this machine — your credentials never leave it.</p>'
            '<form id="inline-login" autocomplete="off">'
            '<label>Username'
            '<input type="text" name="username" required autocomplete="username" autofocus>'
            '</label>'
            '<label>Password'
            '<input type="password" name="password" required autocomplete="current-password">'
            '</label>'
            '<p id="login-error" class="login-error" role="alert"></p>'
            '<button type="submit" class="btn primary">Sign in and install</button>'
            '</form>'
            '</div>'
        )

    body = f"""
    <div class="preview">
      <p class="crumb">Community install · {escape(cat_label)}</p>
      <h1>{name}</h1>
      <p class="description">{description}</p>

      <div class="meta-grid">
        <div><span class="label">Author</span><span class="val">{author}</span></div>
        <div><span class="label">Version</span><span class="val">{version}</span></div>
        <div><span class="label">Requires</span><span class="val">Augmentum &ge; {min_version}</span></div>
        <div><span class="label">License</span><span class="val">{license_str}</span></div>
      </div>

      {f'<div class="tags">{tag_html}</div>' if tag_html else ''}

      <h2>What will change</h2>
      <p>{escape(_change_description(category))}</p>

      <details>
        <summary>Show artifact preview</summary>
        <pre class="artifact"><code>{escape(artifact_json)}</code></pre>
      </details>

      {auth_section}

      <p class="install-status" id="install-status" role="status"></p>
    </div>

    <script>
      (function() {{
        const status = document.getElementById('install-status');
        const installPayload = {{
          manifest_url: {json.dumps(manifest_url)},
          category: {json.dumps(category)},
          artifact: {artifact_json_safe_for_js(artifact_json, artifact)},
          manifest: {json.dumps(manifest, default=str)}
        }};

        const confirmBtn = document.getElementById('confirm-btn');
        if (confirmBtn) {{
          confirmBtn.addEventListener('click', async () => {{
            confirmBtn.disabled = true;
            status.textContent = 'Installing...';
            try {{
              const resp = await fetch('/api/community/install', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(installPayload),
                credentials: 'same-origin'
              }});
              const data = await resp.json().catch(() => ({{}}));
              if (resp.ok) {{
                status.textContent = 'Installed. You can close this tab.';
                confirmBtn.textContent = 'Installed';
              }} else {{
                status.textContent = 'Install failed: ' + (data.detail || resp.statusText);
                confirmBtn.disabled = false;
              }}
            }} catch (err) {{
              status.textContent = 'Install failed: ' + err.message;
              confirmBtn.disabled = false;
            }}
          }});
        }}

        const loginForm = document.getElementById('inline-login');
        if (loginForm) {{
          const loginErr = document.getElementById('login-error');
          const loginBtn = loginForm.querySelector('button[type=submit]');
          loginForm.addEventListener('submit', async (e) => {{
            e.preventDefault();
            loginErr.textContent = '';
            loginBtn.disabled = true;
            loginBtn.textContent = 'Signing in...';
            const fd = new FormData(loginForm);
            try {{
              const resp = await fetch('/api/auth/login', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                  username: fd.get('username'),
                  password: fd.get('password')
                }}),
                credentials: 'same-origin'
              }});
              const data = await resp.json().catch(() => ({{}}));
              if (resp.ok) {{
                // Cookie set — reload to re-enter the handler authenticated.
                window.location.reload();
              }} else {{
                loginErr.textContent = (data.detail || data.message || 'Login failed.');
                loginBtn.disabled = false;
                loginBtn.textContent = 'Sign in and install';
              }}
            }} catch (err) {{
              loginErr.textContent = 'Connection error: ' + err.message;
              loginBtn.disabled = false;
              loginBtn.textContent = 'Sign in and install';
            }}
          }});
        }}
      }})();
    </script>
    """
    return _BASE_PAGE.format(title=escape(name), body=body)


def artifact_json_safe_for_js(artifact_json: str, artifact: Any) -> str:
    """Re-emit the artifact as JSON for the inline script.

    The preview JSON above may be truncated for display; for the actual
    payload we need the full, untruncated artifact.
    """
    return json.dumps(artifact, default=str)


def _category_label(category: str) -> str:
    return {
        "characters": "Character card",
        "reasoning-flows": "Reasoning flow",
        "powers": "Power",
        "knowledge": "Knowledge pack",
    }.get(category, category)


def _change_description(category: str) -> str:
    return {
        "characters": (
            "A new character card will be added to your Characters panel. "
            "No existing data will be modified."
        ),
        "reasoning-flows": (
            "A new reasoning flow will be added to your Reasoning Flows panel. "
            "No existing flows will be modified."
        ),
        "powers": (
            "A new power will be available in your Coder → Powers panel. "
            "Powers do not auto-activate; you must pin them manually."
        ),
        "knowledge": (
            "A knowledge pack will be downloaded from the publisher's hosted "
            "URL and installed into your knowledge directory."
        ),
    }.get(category, "An item will be added to your account.")


# Minimal HTML chrome shared by preview and error pages. Inline styles so
# the page renders correctly even if /ui/ static assets aren't loaded.
_BASE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Augmentum</title>
<style>
  body {{
    background: #111118; color: #ececf1; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    margin: 0; line-height: 1.6;
  }}
  .preview, .error-box {{
    max-width: 36rem; margin: 0 auto; padding: 4rem 1.5rem;
  }}
  h1 {{ font-size: 2rem; margin: 0 0 1rem; font-weight: 500; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.2rem; margin: 2.4rem 0 0.7rem; font-weight: 600; }}
  p {{ margin: 0 0 1rem; color: #a1a1b5; }}
  p.crumb {{ font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; color: #6b6b80; margin-bottom: 1.2rem; }}
  p.description {{ color: #ececf1; opacity: 0.92; font-size: 1.05rem; }}
  .meta-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
    margin: 1.6rem 0; padding: 1rem 1.2rem; background: #1d1d28;
    border: 1px solid #2e2e3c; border-radius: 8px;
  }}
  .meta-grid > div {{ display: flex; flex-direction: column; gap: 0.15rem; }}
  .meta-grid .label {{ font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #6b6b80; }}
  .meta-grid .val {{ color: #ececf1; font-weight: 500; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 1.4rem 0 1rem; }}
  .tag {{ font-size: 0.75rem; color: #a1a1b5; background: #1d1d28; padding: 0.2rem 0.6rem; border-radius: 4px; border: 1px solid #2e2e3c; }}
  pre.artifact {{
    background: #0e0e16; color: #cdd5de; border: 1px solid #2e2e3c;
    border-radius: 6px; padding: 1rem; overflow-x: auto; font-size: 0.85rem;
    line-height: 1.5; max-height: 24rem;
  }}
  details summary {{ cursor: pointer; color: #6c8aff; margin: 1.2rem 0 0.6rem; }}
  .actions {{ display: flex; gap: 0.7rem; margin-top: 2.2rem; flex-wrap: wrap; }}
  .btn {{
    display: inline-block; padding: 0.7rem 1.4rem; border-radius: 8px;
    font-weight: 500; font-size: 0.95rem; text-decoration: none; cursor: pointer;
    border: 1px solid transparent; font-family: inherit;
  }}
  .btn.primary {{ background: #6c8aff; color: #111118; }}
  .btn.primary:hover {{ background: #5a7aef; }}
  .btn.primary:disabled {{ opacity: 0.6; cursor: not-allowed; }}
  .btn.secondary {{ background: transparent; color: #ececf1; border-color: #2e2e3c; }}
  .btn.secondary:hover {{ background: #1d1d28; }}
  .install-status {{ margin-top: 1.2rem; font-size: 0.92rem; color: #a1a1b5; }}
  .signed-in {{ margin-top: 1rem; font-size: 0.88rem; color: #6b6b80; }}
  .signed-in strong {{ color: #ececf1; font-weight: 600; }}
  .login-box {{
    margin: 2rem 0 0; padding: 1.5rem 1.6rem; background: #1d1d28;
    border: 1px solid #2e2e3c; border-radius: 8px;
  }}
  .login-box h2 {{ margin-top: 0; font-size: 1.05rem; }}
  .login-box p {{ font-size: 0.92rem; }}
  .login-box label {{ display: block; margin: 1rem 0 0.5rem; font-size: 0.88rem; color: #a1a1b5; }}
  .login-box input {{
    display: block; width: 100%; margin-top: 0.4rem;
    background: #0e0e16; color: #ececf1; border: 1px solid #2e2e3c;
    border-radius: 6px; padding: 0.7rem 0.85rem; font-size: 0.95rem;
    font-family: inherit;
  }}
  .login-box input:focus {{ outline: 2px solid #6c8aff; outline-offset: 1px; }}
  .login-box button {{ margin-top: 1.4rem; width: 100%; }}
  .login-error {{ color: #f87171; font-size: 0.88rem; min-height: 1.2em; margin: 0.8rem 0 0; }}
  .error-box h1 {{ color: #f87171; }}
  a {{ color: #6c8aff; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
