"""Unit tests for the /api/knowledge/zim serving path.

Covers the link rewriter and HTML safety filter independent of libzim
or the FastAPI route — both are pure functions over strings, fast and
deterministic. The route handler itself is tested separately as part
of the larger knowledge_routes suite when libzim is available.

Why these matter: a regression in the link rewriter sends iframe
navigation back to the live web; a regression in the script stripper
could let upstream JS run inside the user's authenticated origin (the
sandbox is the primary defense, this is belt-and-suspenders).
"""
from __future__ import annotations

import types

from augmentum.proxy.knowledge_routes import (
    _CACHE_ASSET,
    _CACHE_HTML,
    _SKIN_FAMILIES,
    _ZIM_PATH_MAX_LEN,
    _ZIM_REDIRECT_MAX_HOPS,
    _ZIM_THEMES,
    _etag_matches,
    _follow_zim_redirects,
    _neutralize_inline_styles,
    _parse_range_header,
    _refine_mimetype,
    _resolve_zim_path,
    _rewrite_zim_css,
    _rewrite_zim_html,
    _skin_family_for_pack,
    _validate_zim_path,
    _zim_cache_control,
    _zim_etag,
    _zim_reader_styles,
)

# ---------------------------------------------------------------------------
# Link rewriting — internal vs external
# ---------------------------------------------------------------------------

def test_rewrites_relative_href_to_api_path():
    """The core function: ``A/Article_Name`` → ``/api/knowledge/zim/<pack>/A/Article_Name``."""
    html = '<a href="A/Type_2_diabetes">link</a>'
    out = _rewrite_zim_html(html, "mdwiki_en_all_2025-11")
    assert 'href="/api/knowledge/zim/mdwiki_en_all_2025-11/A/Type_2_diabetes"' in out


def test_rewrites_relative_src_for_images():
    html = '<img src="I/photo.png">'
    out = _rewrite_zim_html(html, "wiki_en_2025")
    assert 'src="/api/knowledge/zim/wiki_en_2025/I/photo.png"' in out


def test_preserves_absolute_external_href_but_adds_target_blank():
    """External links must keep their absolute URL (don't rewrite to
    /api/knowledge/zim/...) AND get target="_blank" so they open in a
    new tab instead of trying to navigate the iframe (which Augmentum's
    parent CSP blocks)."""
    html = '<a href="https://www.ncbi.nlm.nih.gov/pmc/article/X">ref</a>'
    out = _rewrite_zim_html(html, "any_pack")
    assert 'href="https://www.ncbi.nlm.nih.gov/pmc/article/X"' in out
    assert 'target="_blank"' in out
    assert 'rel="noopener noreferrer"' in out


def test_preserves_anchor_only_links():
    """Fragment-only links (in-page jumps) shouldn't be rewritten as
    asset paths. The injected <base> tag legitimately contains the
    /api/knowledge/zim/ string, so check the <a> stayed exactly as-is
    rather than assert global absence."""
    html = '<a href="#references">Jump to refs</a>'
    out = _rewrite_zim_html(html, "p")
    assert 'href="#references"' in out
    # The original anchor href should appear unchanged; no rewrite to a
    # /api/knowledge/zim/p/#references-style path.
    assert "/api/knowledge/zim/p/#" not in out
    assert "/api/knowledge/zim/p/references" not in out


def test_preserves_data_uri_src():
    """Data URIs (inline images, etc.) must not get rewritten as paths."""
    html = '<img src="data:image/png;base64,iVBOR">'
    out = _rewrite_zim_html(html, "p")
    assert 'src="data:image/png;base64,iVBOR"' in out


def test_preserves_root_relative_links():
    """``/something`` is already absolute-from-root; don't double-prefix."""
    html = '<a href="/foo">x</a>'
    out = _rewrite_zim_html(html, "p")
    assert 'href="/foo"' in out
    assert "/api/knowledge/zim/p/foo" not in out


def test_rewrites_single_quoted_attributes():
    """MediaWiki occasionally emits single-quoted attribute values."""
    html = "<a href='A/Article'>x</a>"
    out = _rewrite_zim_html(html, "p")
    assert "/api/knowledge/zim/p/A/Article" in out


def test_injects_base_tag_after_head():
    """The base tag is the safety net for any link the regex missed —
    sets the document base so relative URLs resolve to our route."""
    html = "<html><head><meta charset='utf-8'></head><body></body></html>"
    out = _rewrite_zim_html(html, "p_2025")
    assert '<base href="/api/knowledge/zim/p_2025/">' in out
    # And it goes inside <head>
    head_idx = out.find("<head>")
    base_idx = out.find("<base href=")
    assert head_idx < base_idx, "base tag must be inside head"


def test_strips_pre_existing_base_tag():
    """mwoffliner / DevDocs scrapes ship ``<base href="../../">`` in
    the article HTML. Chrome occasionally honors that over ours and
    asset URLs lose the pack_id, producing 404 + MIME-mismatch errors.
    The strip ensures our injected base is the only one in the document."""
    html = '<html><head><base href="../../"><meta charset="utf-8"></head><body></body></html>'
    out = _rewrite_zim_html(html, "p")
    # Article's base is gone
    assert 'href="../../"' not in out
    # Ours remains
    assert '<base href="/api/knowledge/zim/p/">' in out
    # Only ONE base tag in the output
    assert out.count("<base ") == 1


def test_strips_pre_existing_base_tag_self_closing():
    """XHTML-style self-closing variant must also be stripped."""
    html = '<html><head><base href="/" /></head><body></body></html>'
    out = _rewrite_zim_html(html, "p")
    assert 'href="/"' not in out
    assert out.count("<base ") == 1


def test_strips_pre_existing_base_tag_with_attrs():
    """Some scrapes include ``target="_self"`` alongside ``href``."""
    html = '<html><head><base href="/site/" target="_self"></head></html>'
    out = _rewrite_zim_html(html, "p")
    assert 'href="/site/"' not in out
    assert '<base href="/api/knowledge/zim/p/">' in out
    assert out.count("<base ") == 1


def test_marks_html_with_reader_class():
    """Reader-mode CSS uses .augz-reader scoping; the html element gets
    that class added so styles target the right scope."""
    html = "<html><head></head><body>x</body></html>"
    out = _rewrite_zim_html(html, "p")
    assert 'class="augz-reader"' in out


# ---------------------------------------------------------------------------
# Script + handler stripping (defense in depth)
# ---------------------------------------------------------------------------

def test_preserves_script_tags():
    """Scripts and inline handlers are intentionally NOT stripped.
    Modern ZIMs (freeCodeCamp, DevDocs, Stack Exchange mirrors) ship
    as SPAs whose content only renders after their bundle runs. The
    iframe sandbox + CSP is what decides whether scripts execute;
    stripping at the rewriter would break SPAs even when scripts are
    allowed. See the docstring on ``_rewrite_zim_html``."""
    html = "<p>Visible</p><script>boot();</script>"
    out = _rewrite_zim_html(html, "p")
    assert "Visible" in out
    assert "<script>boot();</script>" in out


def test_preserves_inline_event_handlers():
    """``onclick``/``onload`` survive the rewriter for the same SPA
    reason as <script>. The sandbox + CSP layer governs execution."""
    html = '<a href="A/x" onclick="boot()">x</a>'
    out = _rewrite_zim_html(html, "p")
    assert 'onclick="boot()"' in out
    # Internal href still rewritten through our route.
    assert 'href="/api/knowledge/zim/p/A/x"' in out


def test_preserves_inline_handlers_single_quoted():
    """Single-quoted variant — same preservation."""
    html = "<a href='A/x' onmouseover='boot()'>x</a>"
    out = _rewrite_zim_html(html, "p")
    assert "onmouseover='boot()'" in out


# ---------------------------------------------------------------------------
# Nested iframe replacement (NCBI, OWID embed cases)
# ---------------------------------------------------------------------------

def test_replaces_nested_external_iframe_with_link():
    """Articles embed external iframes (NCBI refs, OWID charts) that
    Augmentum's parent CSP rejects as frame-src violations. Replace
    with a small linked card."""
    html = '<iframe src="https://www.ncbi.nlm.nih.gov/pmc/X"></iframe>'
    out = _rewrite_zim_html(html, "p")
    assert "<iframe" not in out
    assert "ncbi.nlm.nih.gov" in out  # placeholder still references the host
    assert 'target="_blank"' in out


def test_keeps_relative_iframe_unchanged():
    """A relative iframe (rare, but possible — embedded ZIM-internal
    content) doesn't need replacing because it would resolve through
    our own route under the same sandbox."""
    html = '<iframe src="A/embedded_chart"></iframe>'
    out = _rewrite_zim_html(html, "p")
    # Implementation detail: relative iframes pass through (the link
    # rewriter handles them like any other src). What matters is no
    # placeholder card replaces them.
    assert "Embedded content from" not in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_handles_html_without_head():
    """Some ZIM entries are HTML fragments (no <head>); the base tag
    still gets injected into a synthesized head."""
    html = '<html><body><p>fragment</p></body></html>'
    out = _rewrite_zim_html(html, "p")
    assert '<base href="/api/knowledge/zim/p/">' in out


def test_handles_empty_html():
    """Defensive: empty input shouldn't crash."""
    out = _rewrite_zim_html("", "p")
    # Some output should exist (at minimum the base tag prefix), no exception.
    assert "/api/knowledge/zim/p/" in out


def test_handles_pack_id_with_special_chars():
    """Pack IDs include hyphens, underscores, and date suffixes —
    the rewriter shouldn't mangle them."""
    html = '<a href="A/x">x</a>'
    out = _rewrite_zim_html(html, "wikipedia_en_simple-2026_02")
    assert 'href="/api/knowledge/zim/wikipedia_en_simple-2026_02/A/x"' in out


# ---------------------------------------------------------------------------
# srcset rewriting (retina / responsive images)
# ---------------------------------------------------------------------------


def test_srcset_dpr_descriptors():
    """``foo 1x, bar 2x`` shape — both URLs rewritten, descriptors kept."""
    html = '<img src="I/a.png" srcset="I/a.png 1x, I/a@2x.png 2x">'
    out = _rewrite_zim_html(html, "p")
    assert "/api/knowledge/zim/p/I/a.png 1x" in out
    assert "/api/knowledge/zim/p/I/a@2x.png 2x" in out


def test_srcset_width_descriptors():
    """``320w / 640w`` shape used by Wikipedia thumb URLs."""
    html = '<img srcset="I/small.jpg 320w, I/large.jpg 640w">'
    out = _rewrite_zim_html(html, "p")
    assert "/api/knowledge/zim/p/I/small.jpg 320w" in out
    assert "/api/knowledge/zim/p/I/large.jpg 640w" in out


def test_srcset_url_only_no_descriptor():
    """A srcset with a single URL and no descriptor is valid HTML."""
    html = '<img srcset="I/a.png">'
    out = _rewrite_zim_html(html, "p")
    assert 'srcset="/api/knowledge/zim/p/I/a.png"' in out


def test_srcset_preserves_absolute_urls():
    """External CDN images are rare but legal — don't rewrite them."""
    html = '<img srcset="https://cdn.example.com/a.png 1x, I/local.png 2x">'
    out = _rewrite_zim_html(html, "p")
    assert "https://cdn.example.com/a.png 1x" in out
    assert "/api/knowledge/zim/p/I/local.png 2x" in out


def test_srcset_protocol_relative_url():
    """``//cdn/img.png`` is a protocol-relative URL — leave it alone."""
    html = '<img srcset="//cdn/img.png 2x, I/local.png 1x">'
    out = _rewrite_zim_html(html, "p")
    assert "//cdn/img.png 2x" in out


def test_srcset_single_quoted():
    """Less common but legal — Wikipedia's parser tests use it."""
    html = "<img srcset='I/a.png 1x, I/b.png 2x'>"
    out = _rewrite_zim_html(html, "p")
    assert "/api/knowledge/zim/p/I/a.png 1x" in out
    assert "/api/knowledge/zim/p/I/b.png 2x" in out


def test_srcset_handles_messy_whitespace():
    """Real Wikipedia output has variable whitespace around commas."""
    html = '<img srcset="  I/a.png   1x ,   I/b.png 2x  ">'
    out = _rewrite_zim_html(html, "p")
    # Both rewritten; descriptors preserved.
    assert "/api/knowledge/zim/p/I/a.png 1x" in out
    assert "/api/knowledge/zim/p/I/b.png 2x" in out


def test_srcset_empty_value_doesnt_crash():
    """Defensive: ``srcset=""`` shouldn't blow up the rewriter."""
    html = '<img srcset="">'
    out = _rewrite_zim_html(html, "p")
    # No exception; output well-formed (srcset stays empty).
    assert 'srcset=""' in out


def test_srcset_does_not_match_data_srcset_attribute():
    """``data-srcset`` is a lazy-load attribute; we want the real
    srcset. Word boundary in the regex prevents accidental hits."""
    # Both attributes present — only `srcset=` should be rewritten;
    # `data-srcset` should pass through (browsers swap it into srcset
    # at JS time and our rewriter would catch it then if needed).
    html = (
        '<img data-srcset="I/lazy.png 1x" '
        'srcset="I/eager.png 1x">'
    )
    out = _rewrite_zim_html(html, "p")
    # Real srcset rewritten
    assert "/api/knowledge/zim/p/I/eager.png 1x" in out
    # data-srcset NOT rewritten (the value contains the lazy URL,
    # which still references the original ZIM-relative path)
    assert 'data-srcset="I/lazy.png 1x"' in out


# ---------------------------------------------------------------------------
# Internal target="_blank" stripping (kiwix-tools #591/#678 parallel)
# ---------------------------------------------------------------------------
#
# Internal links inside ZIM articles sometimes carry ``target="_blank"``
# (Wikipedia portal "main page" links, FANDOM cross-wiki nav, Stack
# Exchange "see more" buttons). Without intervention, clicking one
# opens a bare ``/api/knowledge/zim/{pack}/...`` URL in a new tab —
# outside the browse panel chrome, no back-stack, no AI tools. The
# rewriter strips ``target="_blank"`` from anchors that point at our
# own route while leaving external anchors' ``target="_blank"`` alone
# (those are intended new-tab opens, added by ``_EXTERNAL_ANCHOR_RE``).


def _anchor_attrs_for(href_url: str, html: str) -> str:
    """Return the attribute substring of the <a> tag whose href matches.

    Used to assert "this attribute is/isn't on this anchor" without
    relying on global presence (the injected reader-mode CSS or other
    anchors might also carry the attribute).
    """
    href_idx = html.find(f'href="{href_url}"')
    assert href_idx >= 0, f"href {href_url!r} not found in output"
    tag_start = html.rfind("<a", 0, href_idx)
    tag_end = html.find(">", href_idx)
    return html[tag_start:tag_end + 1]


def test_strips_target_blank_from_internal_anchor_href_first():
    """target="_blank" listed AFTER href on an internal link — stripped."""
    html = '<a href="A/Foo" target="_blank">click</a>'
    out = _rewrite_zim_html(html, "p")
    anchor = _anchor_attrs_for("/api/knowledge/zim/p/A/Foo", out)
    assert 'target="_blank"' not in anchor


def test_strips_target_blank_from_internal_anchor_target_first():
    """target="_blank" listed BEFORE href — also stripped (lookahead
    handles either attribute order)."""
    html = '<a target="_blank" href="A/Foo">click</a>'
    out = _rewrite_zim_html(html, "p")
    anchor = _anchor_attrs_for("/api/knowledge/zim/p/A/Foo", out)
    assert 'target="_blank"' not in anchor


def test_strips_target_blank_with_extra_attrs_in_between():
    """Realistic case: class/id sit between target and href. Both
    orderings handled."""
    html = '<a class="mw-link" target="_blank" id="x" href="A/Foo">click</a>'
    out = _rewrite_zim_html(html, "p")
    anchor = _anchor_attrs_for("/api/knowledge/zim/p/A/Foo", out)
    assert 'target="_blank"' not in anchor
    assert 'class="mw-link"' in anchor  # other attrs untouched
    assert 'id="x"' in anchor


def test_strips_target_blank_single_quoted():
    """ZIMs occasionally emit single-quoted attributes; cover both."""
    html = "<a href='A/Foo' target='_blank'>click</a>"
    out = _rewrite_zim_html(html, "p")
    # The link rewriter normalizes to single quotes for single-quoted
    # source — find the anchor either way.
    sq = out.find("href='/api/knowledge/zim/p/A/Foo'")
    dq = out.find('href="/api/knowledge/zim/p/A/Foo"')
    assert sq >= 0 or dq >= 0
    href_idx = sq if sq >= 0 else dq
    tag_start = out.rfind("<a", 0, href_idx)
    tag_end = out.find(">", href_idx)
    anchor = out[tag_start:tag_end + 1]
    # Match either quote style for the stripped target check
    assert "target='_blank'" not in anchor
    assert 'target="_blank"' not in anchor


def test_keeps_target_blank_on_external_anchor():
    """Regression guard: external anchors keep target="_blank" — that's
    the intended new-tab behavior set by _EXTERNAL_ANCHOR_RE. Stripping
    here would re-break the chrome-error iframe nav we already fixed."""
    html = '<a href="https://example.com">link</a>'
    out = _rewrite_zim_html(html, "p")
    anchor = _anchor_attrs_for("https://example.com", out)
    assert 'target="_blank"' in anchor
    assert 'rel="noopener noreferrer"' in anchor


def test_keeps_non_blank_target_on_internal_anchor():
    """Only target="_blank" gets stripped; other target values
    (target="_self", "_top", a frame name) pass through unchanged.
    Rare in ZIMs but technically allowed; don't accidentally over-strip."""
    html = '<a href="A/Foo" target="_self">click</a>'
    out = _rewrite_zim_html(html, "p")
    anchor = _anchor_attrs_for("/api/knowledge/zim/p/A/Foo", out)
    assert 'target="_self"' in anchor


# ---------------------------------------------------------------------------
# Redirect chain follower
# ---------------------------------------------------------------------------
#
# libzim's Python binding doesn't expose a "follow redirects" flag, so
# the resolver walks the chain explicitly. These tests pin down the
# pass-through, single-hop, multi-hop, cycle, and depth-cap cases.
# Mocked entries use SimpleNamespace to mimic the libzim Entry shape
# (``path``, ``is_redirect``, ``get_redirect_entry``).


def _fake_entry(path, *, is_redirect=False, target=None):
    """Build a libzim-Entry-like object for resolver tests."""
    e = types.SimpleNamespace(path=path, is_redirect=is_redirect)
    if is_redirect:
        e.get_redirect_entry = lambda: target
    return e


def _fake_archive(entries, main_entry=None):
    """Build a libzim-Archive-like object backed by a path-keyed dict."""
    arch = types.SimpleNamespace(main_entry=main_entry)

    def _get(p):
        if p in entries:
            return entries[p]
        # libzim raises a generic exception subclass on miss; the
        # resolver swallows any Exception, so the exact type doesn't
        # matter for the test.
        raise RuntimeError(f"entry not found: {p}")

    arch.get_entry_by_path = _get
    return arch


def test_follow_redirects_passes_terminal_entry_through():
    e = _fake_entry("A/Foo", is_redirect=False)
    assert _follow_zim_redirects(e) is e


def test_follow_redirects_handles_single_hop():
    target = _fake_entry("A/Real", is_redirect=False)
    src = _fake_entry("A/Stub", is_redirect=True, target=target)
    assert _follow_zim_redirects(src) is target


def test_follow_redirects_handles_multi_hop_chain():
    final = _fake_entry("A/Final", is_redirect=False)
    mid = _fake_entry("A/Mid", is_redirect=True, target=final)
    src = _fake_entry("A/Src", is_redirect=True, target=mid)
    assert _follow_zim_redirects(src) is final


def test_follow_redirects_breaks_two_node_cycle():
    """A → B → A → ... must short-circuit, not infinite-loop."""
    a = types.SimpleNamespace(path="A/A", is_redirect=True)
    b = types.SimpleNamespace(path="A/B", is_redirect=True)
    a.get_redirect_entry = lambda: b
    b.get_redirect_entry = lambda: a
    assert _follow_zim_redirects(a) is None


def test_follow_redirects_caps_depth():
    """A chain longer than _ZIM_REDIRECT_MAX_HOPS bails out with None."""
    def chain(n):
        if n == 0:
            return _fake_entry(f"A/E{n}", is_redirect=False)
        return _fake_entry(f"A/E{n}", is_redirect=True, target=chain(n - 1))
    deep = chain(_ZIM_REDIRECT_MAX_HOPS + 2)
    assert _follow_zim_redirects(deep) is None


def test_follow_redirects_returns_none_when_get_redirect_raises():
    """Some malformed ZIMs raise from get_redirect_entry. Resolver
    treats it as a broken chain rather than crashing the request."""
    e = types.SimpleNamespace(path="A/Stub", is_redirect=True)
    e.get_redirect_entry = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert _follow_zim_redirects(e) is None


# ---------------------------------------------------------------------------
# Path resolver — namespace fallback + main-entry fallback + redirects
# ---------------------------------------------------------------------------


def test_resolve_returns_main_entry_for_empty_path():
    main = _fake_entry("A/Index", is_redirect=False)
    arch = _fake_archive({}, main_entry=main)
    assert _resolve_zim_path(arch, "") is main


def test_resolve_returns_main_entry_for_slash_path():
    main = _fake_entry("A/Index", is_redirect=False)
    arch = _fake_archive({}, main_entry=main)
    assert _resolve_zim_path(arch, "/") is main


def test_resolve_follows_main_entry_redirect():
    """Some ZIMs declare main_entry as a redirect — resolver follows
    so /api/knowledge/zim/{pack}/ doesn't return a stub article."""
    real = _fake_entry("A/Index", is_redirect=False)
    stub = _fake_entry("A/Main", is_redirect=True, target=real)
    arch = _fake_archive({}, main_entry=stub)
    assert _resolve_zim_path(arch, "") is real


def test_resolve_finds_path_as_is():
    e = _fake_entry("A/Foo", is_redirect=False)
    arch = _fake_archive({"A/Foo": e})
    assert _resolve_zim_path(arch, "A/Foo") is e


def test_resolve_falls_back_to_A_namespace():
    """Bare path misses, but A/<path> hits — Type-0 article fallback."""
    e = _fake_entry("A/Foo", is_redirect=False)
    arch = _fake_archive({"A/Foo": e})
    assert _resolve_zim_path(arch, "Foo") is e


def test_resolve_falls_back_to_C_namespace():
    """A/<path> misses too, C/<path> hits — Type-1 unified fallback."""
    e = _fake_entry("C/Foo", is_redirect=False)
    arch = _fake_archive({"C/Foo": e})
    assert _resolve_zim_path(arch, "Foo") is e


def test_resolve_skips_prefix_fallback_when_already_prefixed():
    """A path starting with a known namespace shouldn't get re-prefixed
    to A/A/Foo. Direct hit on A/Foo, no double-namespace lookup."""
    a_foo = _fake_entry("A/Foo", is_redirect=False)
    arch = _fake_archive({"A/Foo": a_foo})
    assert _resolve_zim_path(arch, "A/Foo") is a_foo


def test_resolve_returns_none_when_no_candidate_matches():
    arch = _fake_archive({})
    assert _resolve_zim_path(arch, "DoesNotExist") is None


def test_resolve_follows_redirect_on_resolved_entry():
    """The resolved entry itself can be a redirect; resolver walks it."""
    real = _fake_entry("A/Real", is_redirect=False)
    stub = _fake_entry("A/Stub", is_redirect=True, target=real)
    arch = _fake_archive({"A/Stub": stub})
    assert _resolve_zim_path(arch, "A/Stub") is real


def test_resolve_returns_none_when_main_entry_unavailable():
    """Some Type-0 ZIMs lack a main_entry. Resolver doesn't crash —
    returns None so the route emits a 404 instead of 500."""

    class _BrokenArchive:
        @property
        def main_entry(self):
            raise RuntimeError("no main page")

        def get_entry_by_path(self, p):
            raise RuntimeError(f"no entry: {p}")

    assert _resolve_zim_path(_BrokenArchive(), "") is None


def test_resolve_handles_paths_with_slashes_in_them():
    """Stack Exchange ZIMs use paths like questions/12345/title-slug.
    The resolver must accept them as-is on a Type-1 archive."""
    e = _fake_entry("questions/12345/title-slug", is_redirect=False)
    arch = _fake_archive({"questions/12345/title-slug": e})
    assert _resolve_zim_path(arch, "questions/12345/title-slug") is e


# ---------------------------------------------------------------------------
# Path validator (defense in depth before resolution)
# ---------------------------------------------------------------------------


def test_validate_path_passes_normal_entry_path():
    assert _validate_zim_path("A/Type_2_diabetes") == "A/Type_2_diabetes"


def test_validate_path_passes_empty():
    """Empty path is valid — resolver maps it to main_entry."""
    assert _validate_zim_path("") == ""


def test_validate_path_passes_slashed_paths():
    """Stack Exchange / sotoki paths have multiple slashes."""
    assert _validate_zim_path("questions/12345/title") == "questions/12345/title"


def test_validate_path_rejects_dotdot_segment():
    assert _validate_zim_path("../etc/passwd") is None
    assert _validate_zim_path("foo/../bar") is None
    assert _validate_zim_path("foo/bar/..") is None


def test_validate_path_rejects_backslash_traversal():
    """Even on POSIX libzim accepts forward slashes only — but a
    Windows-flavored attacker probe (``foo\\..\\bar``) shouldn't slip
    past the .. check just because the separator is a backslash."""
    assert _validate_zim_path("foo\\..\\bar") is None


def test_validate_path_rejects_leading_slash():
    """Absolute-path probe — also collapses a Type-1 path to look
    like Type-0 sometimes; just reject."""
    assert _validate_zim_path("/etc/passwd") is None


def test_validate_path_rejects_leading_backslash():
    assert _validate_zim_path("\\windows\\system32") is None


def test_validate_path_rejects_nul_byte():
    """Some path-handling code paths truncate at NUL — defensive reject."""
    assert _validate_zim_path("A/Foo\x00.html") is None


def test_validate_path_rejects_control_chars():
    """Newlines / TAB / DEL would corrupt log lines and shouldn't
    appear in legitimate ZIM entry names."""
    assert _validate_zim_path("A/Foo\nBar") is None
    assert _validate_zim_path("A/Foo\tBar") is None
    assert _validate_zim_path("A/Foo\x7fBar") is None


def test_validate_path_rejects_oversized():
    """4 KiB cap prevents pathological resolver work and noisy logs."""
    assert _validate_zim_path("A/" + "x" * (_ZIM_PATH_MAX_LEN + 1)) is None


def test_validate_path_passes_at_max_length():
    """Boundary check — exactly _ZIM_PATH_MAX_LEN should pass."""
    p = "x" * _ZIM_PATH_MAX_LEN
    assert _validate_zim_path(p) == p


# ---------------------------------------------------------------------------
# HTTP Range header parsing (RFC 7233)
# ---------------------------------------------------------------------------


def test_parse_range_no_header_returns_full_200():
    assert _parse_range_header(None, 1000) == (0, 1000, 200)


def test_parse_range_empty_header_returns_full_200():
    assert _parse_range_header("", 1000) == (0, 1000, 200)


def test_parse_range_closed_range():
    """bytes=100-199 → byte 100 through 199 inclusive → [100,200) exclusive."""
    assert _parse_range_header("bytes=100-199", 1000) == (100, 200, 206)


def test_parse_range_open_ended():
    """bytes=500- → from byte 500 to EOF."""
    assert _parse_range_header("bytes=500-", 1000) == (500, 1000, 206)


def test_parse_range_suffix_form():
    """bytes=-500 → last 500 bytes."""
    assert _parse_range_header("bytes=-500", 1000) == (500, 1000, 206)


def test_parse_range_suffix_larger_than_total():
    """RFC 7233 §2.1: suffix > total serves the full representation."""
    assert _parse_range_header("bytes=-2000", 1000) == (0, 1000, 206)


def test_parse_range_full_range():
    """bytes=0- → the whole representation, still 206 because Range
    was specified."""
    assert _parse_range_header("bytes=0-", 1000) == (0, 1000, 206)


def test_parse_range_zero_byte():
    """bytes=0-0 → first byte. Edge of valid; Safari sometimes opens
    media with this to probe Accept-Ranges."""
    assert _parse_range_header("bytes=0-0", 1000) == (0, 1, 206)


def test_parse_range_out_of_bounds_returns_416():
    """start beyond total → 416 with Content-Range: bytes */total
    (the 416 marker; caller composes the header)."""
    assert _parse_range_header("bytes=2000-2999", 1000) == (0, 0, 416)


def test_parse_range_inverted_range_returns_416():
    """end < start is syntactically valid but semantically nonsense."""
    assert _parse_range_header("bytes=500-100", 1000) == (0, 0, 416)


def test_parse_range_negative_suffix_returns_416():
    """bytes=-0 / bytes=- → invalid suffix length."""
    assert _parse_range_header("bytes=-0", 1000) == (0, 0, 416)


def test_parse_range_non_bytes_unit_returns_full_200():
    """Other range units (``items=``, ``rows=``) are valid HTTP but
    not what we serve. RFC 7233: ignore unrecognized units."""
    assert _parse_range_header("items=0-9", 1000) == (0, 1000, 200)


def test_parse_range_malformed_returns_full_200():
    """Garbage Range headers are ignored per RFC 7233 §3.1."""
    assert _parse_range_header("bytes=abc", 1000) == (0, 1000, 200)
    assert _parse_range_header("bytes=", 1000) == (0, 1000, 200)
    assert _parse_range_header("not-a-range", 1000) == (0, 1000, 200)


def test_parse_range_multipart_takes_first_only():
    """Multi-range requests (``bytes=0-99,200-299``) are RFC-legal but
    require multipart/byteranges responses, which are optional. Servers
    serving the first range only and emitting a single 206 are
    compliant; that's what we do."""
    assert _parse_range_header("bytes=0-99,200-299", 1000) == (0, 100, 206)


def test_parse_range_case_insensitive_unit():
    """``Bytes=`` / ``BYTES=`` should match — RFC 7233 unit names
    are case-insensitive."""
    assert _parse_range_header("Bytes=0-99", 1000) == (0, 100, 206)
    assert _parse_range_header("BYTES=0-99", 1000) == (0, 100, 206)


# ---------------------------------------------------------------------------
# MIME extension refinement (octet-stream override)
# ---------------------------------------------------------------------------


def test_refine_mime_passes_legit_types_unchanged():
    """Anything other than octet-stream / empty passes through — we
    don't second-guess a scraper that committed to a real mimetype."""
    assert _refine_mimetype("image/png", "I/photo.svg") == "image/png"
    assert _refine_mimetype("text/html", "A/Foo") == "text/html"
    assert _refine_mimetype("application/x-custom", "A/x") == "application/x-custom"


def test_refine_mime_overrides_octet_stream_for_pdf():
    """Gutenberg case: book stored as octet-stream + .pdf."""
    assert _refine_mimetype("application/octet-stream", "A/book.pdf") == "application/pdf"


def test_refine_mime_overrides_octet_stream_for_epub():
    assert _refine_mimetype(
        "application/octet-stream", "A/book.epub"
    ) == "application/epub+zip"


def test_refine_mime_overrides_octet_stream_for_html():
    """Some legacy ZIMs ship raw HTML as octet-stream — refine so
    the route's HTML branch (rewriter, CSP) still kicks in."""
    assert _refine_mimetype("application/octet-stream", "A/index.html") == "text/html"


def test_refine_mime_overrides_for_image_extensions():
    assert _refine_mimetype("application/octet-stream", "I/img.png") == "image/png"
    assert _refine_mimetype("application/octet-stream", "I/img.svg") == "image/svg+xml"
    assert _refine_mimetype("application/octet-stream", "I/img.webp") == "image/webp"


def test_refine_mime_overrides_for_media():
    """Range/streaming branch only matters if the browser knows it's
    media — refine so audio/video extensions get their proper MIME."""
    assert _refine_mimetype("application/octet-stream", "M/audio.mp3") == "audio/mpeg"
    assert _refine_mimetype("application/octet-stream", "V/clip.webm") == "video/webm"
    assert _refine_mimetype("application/octet-stream", "V/clip.ogv") == "video/ogg"


def test_refine_mime_handles_uppercase_extension():
    """Some scrapers preserve case from the source filesystem; lowercase
    the extension before lookup."""
    assert _refine_mimetype("application/octet-stream", "A/book.PDF") == "application/pdf"


def test_refine_mime_unknown_extension_returns_original():
    """An unknown extension shouldn't blank the mimetype — keep
    octet-stream so the browser falls back to download."""
    assert _refine_mimetype(
        "application/octet-stream", "A/file.xyz"
    ) == "application/octet-stream"


def test_refine_mime_no_extension_returns_original():
    """Wikipedia article paths ('A/Type_2_diabetes') have no extension
    and shouldn't trigger the override."""
    assert _refine_mimetype(
        "application/octet-stream", "A/Type_2_diabetes"
    ) == "application/octet-stream"


def test_refine_mime_empty_input_with_known_ext():
    """Empty mimetype + known extension → infer. Defensive against
    libzim returning '' on a malformed entry."""
    assert _refine_mimetype("", "A/index.html") == "text/html"


def test_refine_mime_empty_input_no_ext_returns_octet():
    """Empty mimetype + no extension → octet-stream (lets the browser
    fall back to download rather than 'unknown' rendering)."""
    assert _refine_mimetype("", "A/Foo") == "application/octet-stream"


# ---------------------------------------------------------------------------
# Cache-Control prediction (used by 304 short-circuit)
# ---------------------------------------------------------------------------


def test_cache_control_html_extension():
    assert _zim_cache_control("A/index.html") == _CACHE_HTML
    assert _zim_cache_control("A/page.HTM") == _CACHE_HTML  # case-insensitive


def test_cache_control_no_extension_treated_as_html():
    """Wikipedia article shape — extensionless paths are HTML in
    practice. Conservative default: short window."""
    assert _zim_cache_control("A/Type_2_diabetes") == _CACHE_HTML
    assert _zim_cache_control("Article") == _CACHE_HTML


def test_cache_control_asset_extensions():
    assert _zim_cache_control("I/photo.png") == _CACHE_ASSET
    assert _zim_cache_control("I/icon.svg") == _CACHE_ASSET
    assert _zim_cache_control("J/bundle.js") == _CACHE_ASSET
    assert _zim_cache_control("M/style.css") == _CACHE_ASSET
    assert _zim_cache_control("V/clip.mp4") == _CACHE_ASSET


def test_cache_control_question_path_no_extension():
    """Stack Exchange shape. No extension on final segment, so HTML."""
    assert _zim_cache_control("questions/12345/title-slug") == _CACHE_HTML


# ---------------------------------------------------------------------------
# ETag generation + If-None-Match matching
# ---------------------------------------------------------------------------


def test_etag_is_quoted_and_stable():
    """RFC 7232 entity-tag is a quoted string. Same inputs ⇒ same tag."""
    a = _zim_etag("p", "A/Foo", "2025-11", 12345, "dark")
    b = _zim_etag("p", "A/Foo", "2025-11", 12345, "dark")
    assert a == b
    assert a.startswith('"') and a.endswith('"')


def test_etag_changes_on_pack_id():
    a = _zim_etag("p1", "A/Foo", "2025-11", 12345, "dark")
    b = _zim_etag("p2", "A/Foo", "2025-11", 12345, "dark")
    assert a != b


def test_etag_changes_on_path():
    a = _zim_etag("p", "A/Foo", "2025-11", 12345, "dark")
    b = _zim_etag("p", "A/Bar", "2025-11", 12345, "dark")
    assert a != b


def test_etag_changes_on_build_date():
    """Pack reinstall busts the cache without manual purge — that's
    the whole point of including build_date in the seed."""
    a = _zim_etag("p", "A/Foo", "2025-10", 12345, "dark")
    b = _zim_etag("p", "A/Foo", "2025-11", 12345, "dark")
    assert a != b


def test_etag_changes_on_mtime():
    """File-level reinstall (catalog re-pull) without metadata change
    still busts cache because mtime moves."""
    a = _zim_etag("p", "A/Foo", "2025-11", 12345, "dark")
    b = _zim_etag("p", "A/Foo", "2025-11", 67890, "dark")
    assert a != b


def test_etag_changes_on_theme():
    """HTML responses are theme-templated; the bytes differ between
    themes so the ETag must too."""
    a = _zim_etag("p", "A/Foo", "2025-11", 12345, "dark")
    b = _zim_etag("p", "A/Foo", "2025-11", 12345, "sepia")
    assert a != b


def test_etag_matches_none_returns_false():
    assert _etag_matches(None, '"abc123"') is False
    assert _etag_matches("", '"abc123"') is False


def test_etag_matches_exact():
    assert _etag_matches('"abc123"', '"abc123"') is True
    assert _etag_matches('"different"', '"abc123"') is False


def test_etag_matches_wildcard():
    """If-None-Match: * matches any existing entity. Browsers don't
    typically send this for GET, but it's valid per RFC 7232."""
    assert _etag_matches("*", '"anything"') is True


def test_etag_matches_comma_separated_list():
    """Browsers cache multiple representations; a list of candidates
    is sent on revalidation."""
    assert _etag_matches('"old", "abc123", "older"', '"abc123"') is True
    assert _etag_matches('"old","abc123","older"', '"abc123"') is True


def test_etag_matches_strips_weak_prefix():
    """Some proxies downgrade strong ETags to weak. Accept either form
    so a downstream cache layer doesn't break our 304 path."""
    assert _etag_matches('W/"abc123"', '"abc123"') is True
    assert _etag_matches('"old", W/"abc123"', '"abc123"') is True


def test_etag_matches_handles_whitespace():
    """Real If-None-Match headers from Chrome/Firefox have inconsistent
    whitespace around commas — match anyway."""
    assert _etag_matches('  "abc123"  ', '"abc123"') is True
    assert _etag_matches('"x" ,  "abc123"  ,  "y"', '"abc123"') is True


# ---------------------------------------------------------------------------
# srcset on <picture><source> — same regex covers both <img> and <source>
# ---------------------------------------------------------------------------
#
# Modern Wikipedia uses ``<picture><source srcset="..."></picture>`` for
# retina assets. The negative lookbehind ``(?<![-\w])`` only blocks the
# ``data-srcset`` form — the space between ``<source`` and ``srcset`` is
# whitespace, which satisfies the lookbehind, so the regex matches.

def test_srcset_inside_picture_source_element():
    """``<picture><source srcset="...">`` — Wikipedia retina pattern."""
    html = (
        '<picture>'
        '<source srcset="I/hires.png 2x" type="image/png">'
        '<img src="I/lores.png" alt="">'
        '</picture>'
    )
    out = _rewrite_zim_html(html, "p")
    # Both <source> and <img> get their paths rewritten.
    assert "/api/knowledge/zim/p/I/hires.png 2x" in out
    assert 'src="/api/knowledge/zim/p/I/lores.png"' in out


def test_srcset_inside_picture_source_multiple_sources():
    """Multiple <source> elements (art-direction) — each rewritten
    independently. media= attribute and other props pass through."""
    html = (
        '<picture>'
        '<source media="(min-width: 1200px)" srcset="I/big.jpg 1200w">'
        '<source media="(min-width: 600px)" srcset="I/mid.jpg 600w">'
        '<img src="I/small.jpg">'
        '</picture>'
    )
    out = _rewrite_zim_html(html, "p")
    assert "/api/knowledge/zim/p/I/big.jpg 1200w" in out
    assert "/api/knowledge/zim/p/I/mid.jpg 600w" in out
    assert 'src="/api/knowledge/zim/p/I/small.jpg"' in out
    # media= queries pass through verbatim
    assert 'media="(min-width: 1200px)"' in out


# ---------------------------------------------------------------------------
# Theme injection — color-scheme + source-order priority
# ---------------------------------------------------------------------------
#
# 2024+ Wikipedia ZIMs (kiwix-js#1376) ship a
# ``@media (prefers-color-scheme: dark)`` block that fights our theme when
# the user's OS is in dark mode and they've selected our light or sepia
# theme. Two defenses:
#   1. ``color-scheme: light`` declares our intent to native UA features
#      (form controls, scrollbars).
#   2. The reader stylesheet is now injected at END of <head> so it wins
#      source-order tie-breaks against the article's own CSS.

def test_color_scheme_declared_for_dark_themes():
    """Dark + midnight palettes set ``color-scheme: dark``."""
    html = '<html><head></head><body>x</body></html>'
    for theme in ("dark", "midnight"):
        out = _rewrite_zim_html(html, "p", theme=theme)
        assert "color-scheme: dark" in out, f"missing for {theme}"


def test_color_scheme_declared_for_light_themes():
    """Light + sepia palettes set ``color-scheme: light``."""
    html = '<html><head></head><body>x</body></html>'
    for theme in ("light", "sepia"):
        out = _rewrite_zim_html(html, "p", theme=theme)
        assert "color-scheme: light" in out, f"missing for {theme}"


def test_color_scheme_unknown_theme_falls_back_to_dark():
    """Unknown theme ID defaults to dark; never raises, never empty."""
    html = '<html><head></head><body>x</body></html>'
    out = _rewrite_zim_html(html, "p", theme="not_a_theme")
    assert "color-scheme: dark" in out


def test_every_theme_has_a_scheme_field():
    """Belt-and-suspenders: ensure ``scheme`` never falls off the
    palette dict in a future refactor."""
    for name, palette in _ZIM_THEMES.items():
        assert "scheme" in palette, f"{name} missing scheme"
        assert palette["scheme"] in ("light", "dark"), name


def test_reader_stylesheet_injected_at_end_of_head():
    """The reader stylesheet must come AFTER the article's own <link>
    and <style> tags so same-specificity ``!important`` rules win on
    source order. Without this, kiwix-js#1376 dark-mode CSS wins."""
    html = (
        '<html><head>'
        '<link rel="stylesheet" href="-/article.css">'
        '<style>body { background: #000 !important }</style>'
        '</head><body>x</body></html>'
    )
    out = _rewrite_zim_html(html, "p", theme="light")
    article_link_idx = out.find('href="/api/knowledge/zim/p/-/article.css"')
    article_style_idx = out.find('background: #000')
    reader_style_idx = out.find('data-augmentum-reader')
    assert article_link_idx > 0
    assert article_style_idx > 0
    assert reader_style_idx > 0
    # Reader style must appear AFTER both the article's <link> and inline <style>.
    assert reader_style_idx > article_link_idx
    assert reader_style_idx > article_style_idx


def test_base_tag_still_at_start_of_head():
    """``<base>`` must come BEFORE any <link>/<img>/<a> so URL resolution
    works for them. Splitting base→start, style→end of head is the
    point of the change."""
    html = (
        '<html><head>'
        '<link rel="stylesheet" href="article.css">'
        '</head><body></body></html>'
    )
    out = _rewrite_zim_html(html, "p")
    base_idx = out.find('<base href="/api/knowledge/zim/p/"')
    link_idx = out.find('<link rel="stylesheet"')
    assert base_idx > 0
    assert link_idx > 0
    assert base_idx < link_idx, "base must precede first <link>"


def test_no_head_tag_falls_back_gracefully():
    """HTML fragments without <head>/</head> still get base + reader
    style injected ahead of body content."""
    html = '<html><body>fragment</body></html>'
    out = _rewrite_zim_html(html, "p")
    assert '<base href="/api/knowledge/zim/p/"' in out
    assert "data-augmentum-reader" in out


def test_head_open_but_no_close_still_injects():
    """Malformed HTML with <head> but no </head> — inject both right
    after <head> as a fallback. Loses source-order priority but better
    than no reader CSS at all."""
    html = '<html><head><body>x</body></html>'
    out = _rewrite_zim_html(html, "p")
    assert '<base href="/api/knowledge/zim/p/"' in out
    assert "data-augmentum-reader" in out


# ---------------------------------------------------------------------------
# CJK font fallbacks — Wikipedia zh/ja/ko ZIMs render proper glyphs
# ---------------------------------------------------------------------------

def test_font_stack_includes_cjk_fallbacks():
    """The reader font stack must include CJK families so Wikipedia
    zh/ja/ko ZIMs don't render as tofu on stripped-down systems where
    the OS default has no CJK coverage."""
    html = '<html><head></head><body>x</body></html>'
    out = _rewrite_zim_html(html, "p", theme="dark")
    # At least one of each region's most-likely-installed family.
    assert "Noto Sans CJK SC" in out      # zh-Hans (Linux/Chrome)
    assert "Noto Sans CJK JP" in out      # ja
    assert "Noto Sans CJK KR" in out      # ko
    assert "PingFang SC" in out           # macOS zh
    assert "Microsoft YaHei" in out       # Windows zh
    assert "Hiragino Sans" in out         # macOS ja
    assert "Yu Gothic" in out             # Windows ja
    assert "Malgun Gothic" in out         # Windows ko


# ---------------------------------------------------------------------------
# CSS url() rewriting — stylesheets served from inside the ZIM
# ---------------------------------------------------------------------------
#
# CSS ``url()`` resolves relative to the stylesheet's own URL, NOT the
# HTML document's <base>. Without rewriting, a stylesheet at
# ``-/style.css`` referencing ``url(I/foo.png)`` resolves to
# ``-/I/foo.png`` (wrong) instead of the ZIM-root ``I/foo.png``. The
# rewriter mirrors the HTML link rewriter's exclusion list.

def test_css_url_unquoted():
    css = ".bg { background: url(I/sprite.png) no-repeat; }"
    out = _rewrite_zim_css(css, "p")
    assert "url(/api/knowledge/zim/p/I/sprite.png)" in out


def test_css_url_double_quoted():
    css = '.bg { background-image: url("I/photo.jpg"); }'
    out = _rewrite_zim_css(css, "p")
    assert 'url("/api/knowledge/zim/p/I/photo.jpg")' in out


def test_css_url_single_quoted():
    css = ".bg { background: url('I/photo.jpg'); }"
    out = _rewrite_zim_css(css, "p")
    assert "url('/api/knowledge/zim/p/I/photo.jpg')" in out


def test_css_url_preserves_absolute_http_url():
    """Absolute URLs (CDN images, fonts) must not be rewritten."""
    css = ".bg { background: url(https://cdn.example.com/foo.png); }"
    out = _rewrite_zim_css(css, "p")
    assert "url(https://cdn.example.com/foo.png)" in out
    assert "/api/knowledge/zim/p/https" not in out


def test_css_url_preserves_protocol_relative():
    css = ".bg { background: url(//cdn.example.com/foo.png); }"
    out = _rewrite_zim_css(css, "p")
    assert "url(//cdn.example.com/foo.png)" in out


def test_css_url_preserves_data_uri():
    """Inline data-URI images (icons, tiny SVGs) — leave alone."""
    css = ".bg { background: url(data:image/png;base64,iVBOR); }"
    out = _rewrite_zim_css(css, "p")
    assert "url(data:image/png;base64,iVBOR)" in out


def test_css_url_preserves_fragment_only():
    """``url(#myFilter)`` references in-document SVG filters."""
    css = ".bg { filter: url(#blur); }"
    out = _rewrite_zim_css(css, "p")
    assert "url(#blur)" in out


def test_css_url_preserves_root_relative():
    """Root-relative paths (``url(/foo.png)``) are excluded for parity
    with the HTML rewriter — fixing those would require a coordinated
    change to both rewriters."""
    css = ".bg { background: url(/I/foo.png); }"
    out = _rewrite_zim_css(css, "p")
    assert "url(/I/foo.png)" in out
    # No double-prefix.
    assert "/api/knowledge/zim/p/I/foo.png" not in out


def test_css_url_with_whitespace_inside_parens():
    """``url(  foo.png  )`` is legal CSS; rewriter handles both
    leading and trailing whitespace."""
    css = ".bg { background: url(  I/spaced.png  ); }"
    out = _rewrite_zim_css(css, "p")
    assert "url(/api/knowledge/zim/p/I/spaced.png)" in out


def test_css_url_idempotent_on_already_rewritten_path():
    """Already-rewritten paths start with ``/`` and are excluded by
    the negative lookahead, so a second pass is a no-op."""
    css = ".bg { background: url(/api/knowledge/zim/p/I/foo.png); }"
    once = _rewrite_zim_css(css, "p")
    twice = _rewrite_zim_css(once, "p")
    assert once == twice
    # Path appears exactly once, not double-prefixed.
    assert once.count("/api/knowledge/zim/p/I/foo.png") == 1


def test_css_url_multiple_in_one_stylesheet():
    """Real stylesheets have many ``url()`` references — each handled."""
    css = (
        "@font-face { src: url(fonts/regular.woff2) format('woff2'); }\n"
        ".icon-a { background: url(I/icon-a.png); }\n"
        ".icon-b { background: url('I/icon-b.png'); }\n"
        ".cdn { background: url(https://cdn/x.png); }\n"
    )
    out = _rewrite_zim_css(css, "p")
    assert "url(/api/knowledge/zim/p/fonts/regular.woff2)" in out
    assert "url(/api/knowledge/zim/p/I/icon-a.png)" in out
    assert "url('/api/knowledge/zim/p/I/icon-b.png')" in out
    assert "url(https://cdn/x.png)" in out  # untouched


def test_css_url_no_matches_returns_input_unchanged():
    """Stylesheet without any ``url()`` calls — return as-is."""
    css = "body { color: red; padding: 10px; }"
    out = _rewrite_zim_css(css, "p")
    assert out == css


def test_css_url_empty_input():
    assert _rewrite_zim_css("", "p") == ""


# ---------------------------------------------------------------------------
# Skin-family detection — pack_id → family
# ---------------------------------------------------------------------------

def test_family_mediawiki_for_known_wiki_packs():
    """All MediaWiki-shaped packs share the same family pack — Vector
    chrome hide + .mw-parser-output layout. The detector must recognize
    the full Wiki* family, not just Wikipedia."""
    for pid in (
        "wikipedia_en_simple-2026_02", "mdwiki_en_all_2025-11",
        "wikibooks_en_all", "wikivoyage_en_all_2025", "wiktionary_en_all",
        "wikiquote_en_all", "wikiversity_en_all", "wikinews_en_all",
        "libretexts_chem", "rationalwiki_en", "appropedia_en",
    ):
        assert _skin_family_for_pack(pid) == "mediawiki", pid


def test_family_devdocs():
    assert _skin_family_for_pack("devdocs_en_python") == "devdocs"
    assert _skin_family_for_pack("devdocs_en_javascript") == "devdocs"


def test_family_stackexchange_covers_se_variants():
    """Stack Exchange ships as ``stack_exchange_*``, ``stackoverflow_*``,
    and sotoki-mirror packs (``sotoki_*``). All three share the SE skin."""
    assert _skin_family_for_pack("stackoverflow_en_all_2025-01") == "stackexchange"
    assert _skin_family_for_pack("stack_exchange_en_codereview") == "stackexchange"
    assert _skin_family_for_pack("sotoki_serverfault") == "stackexchange"


def test_family_freecodecamp():
    assert _skin_family_for_pack("freecodecamp_en_all_2025-03") == "freecodecamp"
    assert _skin_family_for_pack("fcc_en_curriculum") == "freecodecamp"


def test_family_gutenberg_and_wikisource_share_book_skin():
    """Gutenberg + Wikisource both ship bare-HTML books; both serve best
    under the gutenberg family (serif typography, wider measure)."""
    assert _skin_family_for_pack("gutenberg_en_all") == "gutenberg"
    assert _skin_family_for_pack("wikisource_en_all") == "gutenberg"


def test_family_generic_for_unknown_packs():
    """User-imported packs with arbitrary names fall through to generic —
    they still get the baseline (so text is readable) but no chrome hide.
    Empty pack id is treated the same way (defensive)."""
    assert _skin_family_for_pack("") == "generic"
    assert _skin_family_for_pack("my_custom_corpus_v3") == "generic"
    assert _skin_family_for_pack("some_random_zim") == "generic"


def test_family_case_insensitive():
    """Catalog ids usually arrive lowercased, but user-imported names can
    arrive in any case. Detector folds to lower."""
    assert _skin_family_for_pack("Wikipedia_EN_Simple") == "mediawiki"
    assert _skin_family_for_pack("DevDocs_en_python") == "devdocs"


# ---------------------------------------------------------------------------
# Layer 1: baseline always injected
# ---------------------------------------------------------------------------

def test_baseline_palette_vars_present_for_every_family():
    """The ``--augz-*`` palette is the contract between Layer 1 / 2 / 3.
    Every family — including generic — must declare it so the per-family
    pack and inline-style neutralizer can rely on the same tokens."""
    html = "<html><head></head><body><p>hi</p></body></html>"
    for fam in _SKIN_FAMILIES:
        out = _rewrite_zim_html(html, "p", family=fam)
        assert "--augz-bg:" in out, fam
        assert "--augz-text:" in out, fam
        assert "--augz-accent:" in out, fam


def test_baseline_text_color_applied_without_important_on_elements():
    """Per-element text-color rules must NOT use ``!important`` — that
    lets a legitimate inline accent (preserved by the neutralizer when
    it's e.g. a syntax-highlight class) survive. The page-level ``body``
    rule keeps ``!important`` to win against the article's own
    ``body { color: … }`` baseline."""
    css = _zim_reader_styles("generic", "dark")
    # body-level retains !important (article overrides)
    assert "color: var(--augz-text) !important" in css
    # per-element rules use the soft form
    soft = "body p, body li, body td, body th, body dd, body dt"
    assert soft in css


def test_baseline_serves_unknown_packs():
    """Generic family on an unknown pack still applies palette + baseline
    typography. This is the "never invisible text" floor."""
    html = "<html><body><p>obscure content</p></body></html>"
    out = _rewrite_zim_html(html, "weird_unknown_pack", family="generic")
    assert "data-augmentum-reader" in out
    assert "data-augmentum-family=\"generic\"" in out
    assert "--augz-text" in out


# ---------------------------------------------------------------------------
# Layer 2: per-family CSS pack injection
# ---------------------------------------------------------------------------

def test_mediawiki_family_hides_vector_chrome():
    """MediaWiki pack's signature: Vector skin chrome hidden."""
    css = _zim_reader_styles("mediawiki", "dark")
    assert ".vector-header-container" in css
    assert ".mw-parser-output" in css
    # Per-family selectors gated by ``html.augz-reader`` for the header
    # safety scope.
    assert "html.augz-reader body > header" in css


def test_devdocs_family_targets_devdocs_classes():
    """DevDocs pack must target the ``_page``/``_sidebar``/``_app``
    hierarchy — Vector selectors don't help here."""
    css = _zim_reader_styles("devdocs", "dark")
    assert "._page" in css
    assert "._sidebar" in css
    assert "._app" in css
    # Should NOT pull in the mediawiki pack
    assert ".vector-header-container" not in css


def test_stackexchange_family_targets_se_classes():
    css = _zim_reader_styles("stackexchange", "dark")
    assert ".s-card" in css
    assert ".s-prose" in css
    assert ".question" in css and ".answer" in css
    assert ".post-tag" in css


def test_freecodecamp_family_targets_fcc_classes():
    css = _zim_reader_styles("freecodecamp", "dark")
    assert ".challenge-content" in css
    assert ".challenge-editor" in css
    assert ".article-content" in css


def test_gutenberg_family_uses_serif():
    """Books read better in serif — the gutenberg pack is the only family
    that overrides the baseline sans-serif."""
    css = _zim_reader_styles("gutenberg", "dark")
    assert "Georgia" in css
    assert "serif" in css


def test_generic_family_is_baseline_only():
    """Generic = baseline only. No family pack appended."""
    baseline = _zim_reader_styles("generic", "dark")
    # No family-specific selectors leak through
    assert ".mw-parser-output" not in baseline
    assert "._sidebar" not in baseline
    assert ".s-card" not in baseline
    assert ".challenge-content" not in baseline


def test_unknown_family_falls_through_to_baseline_only():
    """Defense in depth: a misspelled family name still produces valid
    CSS (baseline only). No KeyError, no missing palette."""
    css = _zim_reader_styles("not_a_family", "dark")
    assert "--augz-bg" in css
    assert ".mw-parser-output" not in css


def test_family_marker_attribute_set_in_style_tag():
    """The injected ``<style>`` carries ``data-augmentum-family`` so the
    parent frame (or a debugger) can confirm which family rendered."""
    html = "<html><body></body></html>"
    out = _rewrite_zim_html(html, "devdocs_en_python", family="devdocs")
    assert 'data-augmentum-family="devdocs"' in out


# ---------------------------------------------------------------------------
# Layer 3: inline-style color/background neutralizer
# ---------------------------------------------------------------------------

def test_neutralizer_strips_color_declaration():
    html = '<p style="color: #222;">hello</p>'
    out = _neutralize_inline_styles(html)
    assert "color:" not in out
    assert "<p>" in out  # empty style attr dropped


def test_neutralizer_strips_background_color():
    html = '<div style="background-color: #fff; padding: 8px;">x</div>'
    out = _neutralize_inline_styles(html)
    assert "background-color:" not in out
    # Layout property must survive
    assert "padding: 8px" in out


def test_neutralizer_strips_background_shorthand():
    """``background:`` shorthand includes color — has to be stripped."""
    html = '<span style="background: red; margin: 4px;">y</span>'
    out = _neutralize_inline_styles(html)
    assert "background:" not in out
    assert "margin: 4px" in out


def test_neutralizer_leaves_layout_properties():
    """Non-color declarations must pass through untouched."""
    html = '<div style="width: 50%; margin: 1em; display: flex;">z</div>'
    out = _neutralize_inline_styles(html)
    assert "width: 50%" in out
    assert "margin: 1em" in out
    assert "display: flex" in out


def test_neutralizer_drops_empty_style_attr():
    """Style attr with only color → wholly dropped (clean output)."""
    html = '<p style="color:red">x</p>'
    out = _neutralize_inline_styles(html)
    assert "style=" not in out
    assert "<p>x</p>" in out


def test_neutralizer_handles_multiple_decls():
    """Multiple color decls on one element — all stripped, layout kept."""
    html = '<td style="color: white; background: black; padding: 4px; width: 100px;">cell</td>'
    out = _neutralize_inline_styles(html)
    assert "color:" not in out
    assert "background:" not in out
    assert "padding: 4px" in out
    assert "width: 100px" in out


def test_neutralizer_is_idempotent():
    html = '<p style="color: red; margin: 1em;">x</p>'
    once = _neutralize_inline_styles(html)
    twice = _neutralize_inline_styles(once)
    assert once == twice


def test_neutralizer_handles_single_quoted_style():
    html = "<p style='color: red; margin: 1em;'>x</p>"
    out = _neutralize_inline_styles(html)
    assert "color:" not in out
    assert "margin: 1em" in out


def test_neutralizer_runs_for_reader_mode_on():
    """The full pipeline strips hostile inline styles when reader is on."""
    html = '<html><body><p style="color:#222">bad on dark</p></body></html>'
    out = _rewrite_zim_html(html, "p", family="generic", reader_mode=True)
    assert 'style="color:#222"' not in out
    assert 'style="color: #222"' not in out


def test_neutralizer_skipped_for_raw_mode():
    """Raw mode preserves the user's original inline styles."""
    html = '<html><body><p style="color:#222">original look</p></body></html>'
    out = _rewrite_zim_html(html, "p", family="generic", reader_mode=False)
    # The original style attr is preserved verbatim.
    assert 'style="color:#222"' in out


# ---------------------------------------------------------------------------
# Reader-off escape hatch
# ---------------------------------------------------------------------------

def test_raw_mode_skips_style_injection():
    """No ``<style data-augmentum-reader>`` block when raw mode is on."""
    html = "<html><head></head><body>raw</body></html>"
    out = _rewrite_zim_html(html, "p", family="mediawiki", reader_mode=False)
    assert "data-augmentum-reader" not in out
    # Confirm the augz-reader class is STILL set — that's a DOM hook for
    # the parent, not gated on reader mode.
    assert 'class="augz-reader"' in out


def test_raw_mode_still_rewrites_links():
    """Link rewriting is functional, not cosmetic — must run in raw mode
    or internal navigation escapes the iframe to the live web."""
    html = '<html><body><a href="A/Foo">link</a></body></html>'
    out = _rewrite_zim_html(html, "wp", family="mediawiki", reader_mode=False)
    assert 'href="/api/knowledge/zim/wp/A/Foo"' in out


def test_raw_mode_still_injects_base():
    """The injected ``<base>`` is functional (relative URL resolution)
    and must survive raw mode."""
    html = "<html><head></head><body></body></html>"
    out = _rewrite_zim_html(html, "wp", family="mediawiki", reader_mode=False)
    assert '<base href="/api/knowledge/zim/wp/">' in out


def test_reader_mode_default_is_on():
    """Default keyword arg keeps existing call sites' behavior — style
    injection happens when ``reader_mode`` isn't passed."""
    html = "<html><body></body></html>"
    out = _rewrite_zim_html(html, "p")
    assert "data-augmentum-reader" in out


# ---------------------------------------------------------------------------
# ETag must include family + reader_mode
# ---------------------------------------------------------------------------

def test_etag_changes_with_family():
    """Cache must bust when the family changes — a re-classification or
    detector tweak otherwise serves stale Vector CSS to a DevDocs page."""
    base_args = ("p", "A/Foo", "2026-01-01", 12345, "dark")
    mw = _zim_etag(*base_args, family="mediawiki")
    dd = _zim_etag(*base_args, family="devdocs")
    assert mw != dd


def test_etag_changes_with_reader_mode():
    """Reader-on and raw responses are different byte streams — they
    must have different ETags."""
    base_args = ("p", "A/Foo", "2026-01-01", 12345, "dark")
    on = _zim_etag(*base_args, family="mediawiki", reader_mode=True)
    off = _zim_etag(*base_args, family="mediawiki", reader_mode=False)
    assert on != off


def test_etag_stable_for_same_inputs():
    """Same inputs → same ETag (determinism check; otherwise If-None-Match
    revalidation never short-circuits)."""
    args = ("p", "A/Foo", "2026-01-01", 12345, "dark")
    assert (
        _zim_etag(*args, family="devdocs", reader_mode=True)
        == _zim_etag(*args, family="devdocs", reader_mode=True)
    )
